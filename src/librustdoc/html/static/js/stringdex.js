// ignore-tidy-filelength
/**
 * @import * as stringdex from "./stringdex.d.ts"
 */

const EMPTY_UINT8 = new Uint8Array();

/**
 * @property {Uint8Array} keysAndCardinalities
 * @property {Uint8Array[]} containers
 */
class RoaringBitmap {
    /**
     * @param {Uint8Array|null} u8array
     * @param {number} [startingOffset]
    */
    constructor(u8array, startingOffset) {
        const start = startingOffset ? startingOffset : 0;
        let i = start;
        /** @type {Uint8Array} */
        this.keysAndCardinalities = EMPTY_UINT8;
        /** @type {(RoaringBitmapArray|RoaringBitmapBits|RoaringBitmapRun)[]} */
        this.containers = [];
        /** @type {number} */
        this.consumed_len_bytes = 0;
        if (u8array === null || u8array.length === i || u8array[i] === 0) {
            return this;
        } else if (u8array[i] > 0xf0) {
            // Special representation of tiny sets that are close together
            const lspecial = u8array[i] & 0x0f;
            this.keysAndCardinalities = new Uint8Array(lspecial * 4);
            let pspecial = i + 1;
            let key = u8array[pspecial + 2] | (u8array[pspecial + 3] << 8);
            let value = u8array[pspecial] | (u8array[pspecial + 1] << 8);
            let entry = (key << 16) | value;
            let container;
            container = new RoaringBitmapArray(1, new Uint8Array(4));
            container.array[0] = value & 0xFF;
            container.array[1] = (value >> 8) & 0xFF;
            this.containers.push(container);
            this.keysAndCardinalities[0] = key;
            this.keysAndCardinalities[1] = key >> 8;
            pspecial += 4;
            for (let ispecial = 1; ispecial < lspecial; ispecial += 1) {
                entry += u8array[pspecial] | (u8array[pspecial + 1] << 8);
                value = entry & 0xffff;
                key = entry >> 16;
                container = this.addToArrayAt(key);
                const cardinalityOld = container.cardinality;
                container.array[cardinalityOld * 2] = value & 0xFF;
                container.array[(cardinalityOld * 2) + 1] = (value >> 8) & 0xFF;
                container.cardinality = cardinalityOld + 1;
                pspecial += 2;
            }
            this.consumed_len_bytes = pspecial - i;
            return this;
        } else if (u8array[i] < 0x3a) {
            // Special representation of tiny sets with arbitrary 32-bit integers
            const lspecial = u8array[i];
            this.keysAndCardinalities = new Uint8Array(lspecial * 4);
            let pspecial = i + 1;
            for (let ispecial = 0; ispecial < lspecial; ispecial += 1) {
                const key = u8array[pspecial + 2] | (u8array[pspecial + 3] << 8);
                const value = u8array[pspecial] | (u8array[pspecial + 1] << 8);
                const container = this.addToArrayAt(key);
                const cardinalityOld = container.cardinality;
                container.array[cardinalityOld * 2] = value & 0xFF;
                container.array[(cardinalityOld * 2) + 1] = (value >> 8) & 0xFF;
                container.cardinality = cardinalityOld + 1;
                pspecial += 4;
            }
            this.consumed_len_bytes = pspecial - i;
            return this;
        }
        // https://github.com/RoaringBitmap/RoaringFormatSpec
        //
        // Roaring bitmaps are used for flags that can be kept in their
        // compressed form, even when loaded into memory. This decoder
        // turns the containers into objects, but uses byte array
        // slices of the original format for the data payload.
        const has_runs = u8array[i] === 0x3b;
        if (u8array[i] !== 0x3a && u8array[i] !== 0x3b) {
            throw new Error("not a roaring bitmap: " + u8array[i]);
        }
        const size = has_runs ?
            ((u8array[i + 2] | (u8array[i + 3] << 8)) + 1) :
            ((u8array[i + 4] | (u8array[i + 5] << 8) |
             (u8array[i + 6] << 16) | (u8array[i + 7] << 24)));
        i += has_runs ? 4 : 8;
        let is_run;
        if (has_runs) {
            const is_run_len = (size + 7) >> 3;
            is_run = new Uint8Array(u8array.buffer, i + u8array.byteOffset, is_run_len);
            i += is_run_len;
        } else {
            is_run = EMPTY_UINT8;
        }
        this.keysAndCardinalities = u8array.subarray(i, i + (size * 4));
        i += size * 4;
        let offsets = null;
        if (!has_runs || size >= 4) {
            offsets = [];
            for (let j = 0; j < size; ++j) {
                offsets.push(u8array[i] | (u8array[i + 1] << 8) | (u8array[i + 2] << 16) |
                    (u8array[i + 3] << 24));
                i += 4;
            }
        }
        for (let j = 0; j < size; ++j) {
            if (offsets && offsets[j] !== i - start) {
                throw new Error(`corrupt bitmap ${j}: ${i - start} / ${offsets[j]}`);
            }
            const cardinality = (this.keysAndCardinalities[(j * 4) + 2] |
                (this.keysAndCardinalities[(j * 4) + 3] << 8)) + 1;
            if (is_run[j >> 3] & (1 << (j & 0x7))) {
                const runcount = (u8array[i] | (u8array[i + 1] << 8));
                i += 2;
                this.containers.push(new RoaringBitmapRun(
                    runcount,
                    new Uint8Array(u8array.buffer, i + u8array.byteOffset, runcount * 4),
                ));
                i += runcount * 4;
            } else if (cardinality >= 4096) {
                this.containers.push(new RoaringBitmapBits(new Uint8Array(
                    u8array.buffer,
                    i + u8array.byteOffset, 8192,
                )));
                i += 8192;
            } else {
                const end = cardinality * 2;
                this.containers.push(new RoaringBitmapArray(
                    cardinality,
                    new Uint8Array(u8array.buffer, i + u8array.byteOffset, end),
                ));
                i += end;
            }
        }
        this.consumed_len_bytes = i - start;
    }
    /**
     * @param {number} number
     * @returns {RoaringBitmap}
     */
    static makeSingleton(number) {
        const result = new RoaringBitmap(null, 0);
        result.keysAndCardinalities = Uint8Array.of(
            (number >> 16), (number >> 24),
            0, 0, // keysAndCardinalities stores the true cardinality minus 1
        );
        result.containers.push(new RoaringBitmapArray(
            1,
            Uint8Array.of(number, number >> 8),
        ));
        return result;
    }
    /** @returns {RoaringBitmap} */
    static everything() {
        if (EVERYTHING_BITMAP.isEmpty()) {
            let i = 0;
            const l = 1 << 16;
            const everything_range = new RoaringBitmapRun(1, Uint8Array.of(0, 0, 0xff, 0xff));
            EVERYTHING_BITMAP.keysAndCardinalities = new Uint8Array(l * 4);
            while (i < l) {
                EVERYTHING_BITMAP.containers.push(everything_range);
                // key
                EVERYTHING_BITMAP.keysAndCardinalities[(i * 4) + 0] = i;
                EVERYTHING_BITMAP.keysAndCardinalities[(i * 4) + 1] = i >> 8;
                // cardinality (minus one)
                EVERYTHING_BITMAP.keysAndCardinalities[(i * 4) + 2] = 0xff;
                EVERYTHING_BITMAP.keysAndCardinalities[(i * 4) + 3] = 0xff;
                i += 1;
            }
        }
        return EVERYTHING_BITMAP;
    }
    /** @returns {RoaringBitmap} */
    static empty() {
        return EMPTY_BITMAP;
    }
    /** @returns {boolean} */
    isEmpty() {
        return this.containers.length === 0;
    }
    /**
     * Helper function used when constructing bitmaps from lists.
     * Returns an array container with at least two free byte slots
     * and bumps `this.cardinalities`.
     * @param {number} key
     * @returns {RoaringBitmapArray}
     */
    addToArrayAt(key) {
        let mid = this.getContainerId(key);
        /** @type {RoaringBitmapArray|RoaringBitmapBits|RoaringBitmapRun} */
        let container;
        if (mid === -1) {
            container = new RoaringBitmapArray(0, new Uint8Array(2));
            mid = this.containers.length;
            this.containers.push(container);
            if (mid * 4 > this.keysAndCardinalities.length) {
                const keysAndContainers = new Uint8Array(mid * 8);
                keysAndContainers.set(this.keysAndCardinalities);
                this.keysAndCardinalities = keysAndContainers;
            }
            this.keysAndCardinalities[(mid * 4) + 0] = key;
            this.keysAndCardinalities[(mid * 4) + 1] = key >> 8;
        } else {
            container = this.containers[mid];
            const cardinalityOld =
                this.keysAndCardinalities[(mid * 4) + 2] |
                (this.keysAndCardinalities[(mid * 4) + 3] << 8);
            const cardinality = cardinalityOld + 1;
            this.keysAndCardinalities[(mid * 4) + 2] = cardinality;
            this.keysAndCardinalities[(mid * 4) + 3] = cardinality >> 8;
        }
        // the logic for handing this number is annoying, because keysAndCardinalities stores
        // the cardinality *minus one*, so that it can count up to 65536 with only two bytes
        // (because empty containers are never stored).
        //
        // So, if this is a new container, the stored cardinality contains `0 0`, which is
        // the proper value of the old cardinality (an imaginary empty container existed).
        // If this is adding to an existing container, then the above `else` branch bumps it
        // by one, leaving us with a proper value of `cardinality - 1`.
        const cardinalityOld =
            this.keysAndCardinalities[(mid * 4) + 2] |
            (this.keysAndCardinalities[(mid * 4) + 3] << 8);
        if (!(container instanceof RoaringBitmapArray) ||
            container.array.byteLength < ((cardinalityOld + 1) * 2)
        ) {
            const newBuf = new Uint8Array((cardinalityOld + 1) * 4);
            let idx = 0;
            for (const cvalue of container.values()) {
                newBuf[idx] = cvalue & 0xFF;
                newBuf[idx + 1] = (cvalue >> 8) & 0xFF;
                idx += 2;
            }
            if (container instanceof RoaringBitmapArray) {
                container.cardinality = cardinalityOld;
                container.array = newBuf;
                return container;
            }
            const newcontainer = new RoaringBitmapArray(cardinalityOld, newBuf);
            this.containers[mid] = newcontainer;
            return newcontainer;
        } else {
            return container;
        }
    }
    /**
     * @param {RoaringBitmap} that
     * @returns {RoaringBitmap}
     */
    union(that) {
        if (this.isEmpty()) {
            return that;
        }
        if (that.isEmpty()) {
            return this;
        }
        if (this === RoaringBitmap.everything() || that === RoaringBitmap.everything()) {
            return RoaringBitmap.everything();
        }
        let i = 0;
        const il = this.containers.length;
        let j = 0;
        const jl = that.containers.length;
        const result = new RoaringBitmap(null, 0);
        result.keysAndCardinalities = new Uint8Array((il + jl) * 4);
        while (i < il || j < jl) {
            const ik = i * 4;
            const jk = j * 4;
            const k = result.containers.length * 4;
            if (j >= jl || (i < il && (
                (this.keysAndCardinalities[ik + 1] < that.keysAndCardinalities[jk + 1]) ||
                (this.keysAndCardinalities[ik + 1] === that.keysAndCardinalities[jk + 1] &&
                    this.keysAndCardinalities[ik] < that.keysAndCardinalities[jk])
            ))) {
                result.keysAndCardinalities[k + 0] = this.keysAndCardinalities[ik + 0];
                result.keysAndCardinalities[k + 1] = this.keysAndCardinalities[ik + 1];
                result.keysAndCardinalities[k + 2] = this.keysAndCardinalities[ik + 2];
                result.keysAndCardinalities[k + 3] = this.keysAndCardinalities[ik + 3];
                result.containers.push(this.containers[i]);
                i += 1;
            } else if (i >= il || (j < jl && (
                (that.keysAndCardinalities[jk + 1] < this.keysAndCardinalities[ik + 1]) ||
                (that.keysAndCardinalities[jk + 1] === this.keysAndCardinalities[ik + 1] &&
                    that.keysAndCardinalities[jk] < this.keysAndCardinalities[ik])
            ))) {
                result.keysAndCardinalities[k + 0] = that.keysAndCardinalities[jk + 0];
                result.keysAndCardinalities[k + 1] = that.keysAndCardinalities[jk + 1];
                result.keysAndCardinalities[k + 2] = that.keysAndCardinalities[jk + 2];
                result.keysAndCardinalities[k + 3] = that.keysAndCardinalities[jk + 3];
                result.containers.push(that.containers[j]);
                j += 1;
            } else {
                // this key is not smaller than that key
                // that key is not smaller than this key
                // they must be equal
                const thisContainer = this.containers[i];
                const thatContainer = that.containers[j];
                let card = 0;
                if (thisContainer instanceof RoaringBitmapBits &&
                    thatContainer instanceof RoaringBitmapBits
                ) {
                    const resultArray = new Uint8Array(
                        thisContainer.array.length > thatContainer.array.length ?
                            thisContainer.array.length :
                            thatContainer.array.length,
                    );
                    let k = 0;
                    const kl = resultArray.length;
                    while (k < kl) {
                        const c = thisContainer.array[k] | thatContainer.array[k];
                        resultArray[k] = c;
                        card += bitCount(c);
                        k += 1;
                    }
                    result.containers.push(new RoaringBitmapBits(resultArray));
                } else {
                    const thisValues = thisContainer.values();
                    const thatValues = thatContainer.values();
                    let thisResult = thisValues.next();
                    let thatResult = thatValues.next();
                    /** @type {Array<number>} */
                    const resultValues = [];
                    while (!thatResult.done || !thisResult.done) {
                        // generator will definitely implement the iterator protocol correctly
                        /** @type {number} */
                        const thisValue = thisResult.value;
                        /** @type {number} */
                        const thatValue = thatResult.value;
                        if (thatResult.done || thisValue < thatValue) {
                            resultValues.push(thisValue);
                            thisResult = thisValues.next();
                        } else if (thisResult.done || thatValue < thisValue) {
                            resultValues.push(thatValue);
                            thatResult = thatValues.next();
                        } else {
                            // this value is not smaller than that value
                            // that value is not smaller than this value
                            // they must be equal
                            resultValues.push(thisValue);
                            thisResult = thisValues.next();
                            thatResult = thatValues.next();
                        }
                    }
                    const resultArray = new Uint8Array(resultValues.length * 2);
                    let k = 0;
                    for (const value of resultValues) {
                        // roaring bitmap is little endian
                        resultArray[k] = value & 0xFF;
                        resultArray[k + 1] = (value >> 8) & 0xFF;
                        k += 2;
                    }
                    result.containers.push(new RoaringBitmapArray(
                        resultValues.length,
                        resultArray,
                    ));
                    card = resultValues.length;
                }
                result.keysAndCardinalities[k + 0] = this.keysAndCardinalities[ik + 0];
                result.keysAndCardinalities[k + 1] = this.keysAndCardinalities[ik + 1];
                card -= 1;
                result.keysAndCardinalities[k + 2] = card;
                result.keysAndCardinalities[k + 3] = card >> 8;
                i += 1;
                j += 1;
            }
        }
        return result;
    }
    /**
     * @param {RoaringBitmap} that
     * @returns {RoaringBitmap}
     */
    intersection(that) {
        if (this.isEmpty() || that.isEmpty()) {
            return EMPTY_BITMAP;
        }
        if (this === RoaringBitmap.everything()) {
            return that;
        }
        if (that === RoaringBitmap.everything()) {
            return this;
        }
        let i = 0;
        const il = this.containers.length;
        let j = 0;
        const jl = that.containers.length;
        const result = new RoaringBitmap(null, 0);
        result.keysAndCardinalities = new Uint8Array((il > jl ? il : jl) * 4);
        while (i < il && j < jl) {
            const ik = i * 4;
            const jk = j * 4;
            const k = result.containers.length * 4;
            if (j >= jl || (i < il && (
                (this.keysAndCardinalities[ik + 1] < that.keysAndCardinalities[jk + 1]) ||
                (this.keysAndCardinalities[ik + 1] === that.keysAndCardinalities[jk + 1] &&
                    this.keysAndCardinalities[ik] < that.keysAndCardinalities[jk])
            ))) {
                i += 1;
            } else if (i >= il || (j < jl && (
                (that.keysAndCardinalities[jk + 1] < this.keysAndCardinalities[ik + 1]) ||
                (that.keysAndCardinalities[jk + 1] === this.keysAndCardinalities[ik + 1] &&
                    that.keysAndCardinalities[jk] < this.keysAndCardinalities[ik])
            ))) {
                j += 1;
            } else {
                // this key is not smaller than that key
                // that key is not smaller than this key
                // they must be equal
                const thisContainer = this.containers[i];
                const thatContainer = that.containers[j];
                let card = 0;
                if (thisContainer instanceof RoaringBitmapBits &&
                    thatContainer instanceof RoaringBitmapBits
                ) {
                    const resultArray = new Uint8Array(
                        thisContainer.array.length > thatContainer.array.length ?
                            thisContainer.array.length :
                            thatContainer.array.length,
                    );
                    let k = 0;
                    const kl = resultArray.length;
                    while (k < kl) {
                        const c = thisContainer.array[k] & thatContainer.array[k];
                        resultArray[k] = c;
                        card += bitCount(c);
                        k += 1;
                    }
                    if (card !== 0) {
                        result.containers.push(new RoaringBitmapBits(resultArray));
                    }
                } else {
                    const thisValues = thisContainer.values();
                    const thatValues = thatContainer.values();
                    let thisValue = thisValues.next();
                    let thatValue = thatValues.next();
                    const resultValues = [];
                    while (!thatValue.done && !thisValue.done) {
                        if (thisValue.value < thatValue.value) {
                            thisValue = thisValues.next();
                        } else if (thatValue.value < thisValue.value) {
                            thatValue = thatValues.next();
                        } else {
                            // this value is not smaller than that value
                            // that value is not smaller than this value
                            // they must be equal
                            resultValues.push(thisValue.value);
                            thisValue = thisValues.next();
                            thatValue = thatValues.next();
                        }
                    }
                    card = resultValues.length;
                    if (card !== 0) {
                        const resultArray = new Uint8Array(resultValues.length * 2);
                        let k = 0;
                        for (const value of resultValues) {
                            // roaring bitmap is little endian
                            resultArray[k] = value & 0xFF;
                            resultArray[k + 1] = (value >> 8) & 0xFF;
                            k += 2;
                        }
                        result.containers.push(new RoaringBitmapArray(
                            resultValues.length,
                            resultArray,
                        ));
                    }
                }
                if (card !== 0) {
                    result.keysAndCardinalities[k + 0] = this.keysAndCardinalities[ik + 0];
                    result.keysAndCardinalities[k + 1] = this.keysAndCardinalities[ik + 1];
                    card -= 1;
                    result.keysAndCardinalities[k + 2] = card;
                    result.keysAndCardinalities[k + 3] = card >> 8;
                }
                i += 1;
                j += 1;
            }
        }
        return result;
    }
    /** @param {number} keyvalue */
    contains(keyvalue) {
        const key = keyvalue >> 16;
        const value = keyvalue & 0xFFFF;
        const mid = this.getContainerId(key);
        return mid === -1 ? false : this.containers[mid].contains(value);
    }
    /**
     * @param {number} keyvalue
     * @returns {RoaringBitmap}
     */
    remove(keyvalue) {
        const key = keyvalue >> 16;
        const value = keyvalue & 0xFFFF;
        const mid = this.getContainerId(key);
        if (mid === -1) {
            return this;
        }
        const container = this.containers[mid];
        if (!container.contains(value)) {
            return this;
        }
        const newCardinality = (this.keysAndCardinalities[(mid * 4) + 2] |
            (this.keysAndCardinalities[(mid * 4) + 3] << 8));
        const l = this.containers.length;
        const m = l - (newCardinality === 0 ? 1 : 0);
        const result = new RoaringBitmap(null, 0);
        result.keysAndCardinalities = new Uint8Array(m * 4);
        let j = 0;
        for (let i = 0; i < l; i += 1) {
            if (i === mid) {
                if (newCardinality !== 0) {
                    result.keysAndCardinalities[(j * 4) + 0] = key;
                    result.keysAndCardinalities[(j * 4) + 1] = key >> 8;
                    const card = newCardinality - 1;
                    result.keysAndCardinalities[(j * 4) + 2] = card;
                    result.keysAndCardinalities[(j * 4) + 3] = card >> 8;
                    const newContainer = new RoaringBitmapArray(
                        newCardinality,
                        new Uint8Array(newCardinality * 2),
                    );
                    let newContainerSlot = 0;
                    for (const containerValue of container.values()) {
                        if (containerValue !== value) {
                            newContainer.array[newContainerSlot] = value & 0xFF;
                            newContainerSlot += 1;
                            newContainer.array[newContainerSlot] = value >> 8;
                            newContainerSlot += 1;
                        }
                    }
                    result.containers.push(newContainer);
                    j += 1;
                }
            } else {
                result.keysAndCardinalities[(j * 4) + 0] = this.keysAndCardinalities[(i * 4) + 0];
                result.keysAndCardinalities[(j * 4) + 1] = this.keysAndCardinalities[(i * 4) + 1];
                result.keysAndCardinalities[(j * 4) + 2] = this.keysAndCardinalities[(i * 4) + 2];
                result.keysAndCardinalities[(j * 4) + 3] = this.keysAndCardinalities[(i * 4) + 3];
                result.containers.push(this.containers[i]);
                j += 1;
            }
        }
        return result;
    }
    /**
     * @param {number} key
     * @returns {number}
     */
    getContainerId(key) {
        // Binary search algorithm copied from
        // https://en.wikipedia.org/wiki/Binary_search#Procedure
        //
        // Format is required by specification to be sorted.
        // Because keys are 16 bits and unique, length can't be
        // bigger than 2**16, and because we have 32 bits of safe int,
        // left + right can't overflow.
        let left = 0;
        let right = this.containers.length - 1;
        while (left <= right) {
            const mid = Math.floor((left + right) / 2);
            const x = this.keysAndCardinalities[(mid * 4)] |
                (this.keysAndCardinalities[(mid * 4) + 1] << 8);
            if (x < key) {
                left = mid + 1;
            } else if (x > key) {
                right = mid - 1;
            } else {
                return mid;
            }
        }
        return -1;
    }
    * entries() {
        const l = this.containers.length;
        for (let i = 0; i < l; ++i) {
            const key = this.keysAndCardinalities[i * 4] |
                (this.keysAndCardinalities[(i * 4) + 1] << 8);
            for (const value of this.containers[i].values()) {
                yield (key << 16) | value;
            }
        }
    }
    /**
     * @returns {number|null}
     */
    first() {
        for (const entry of this.entries()) {
            return entry;
        }
        return null;
    }
    /**
     * @returns {number}
     */
    cardinality() {
        let result = 0;
        const l = this.containers.length;
        for (let i = 0; i < l; ++i) {
            const card = this.keysAndCardinalities[(i * 4) + 2] |
                (this.keysAndCardinalities[(i * 4) + 3] << 8);
            result += card + 1;
        }
        return result;
    }
}

class RoaringBitmapRun {
    /**
     * @param {number} runcount
     * @param {Uint8Array} array
     */
    constructor(runcount, array) {
        this.runcount = runcount;
        this.array = array;
    }
    /** @param {number} value */
    contains(value) {
        // Binary search algorithm copied from
        // https://en.wikipedia.org/wiki/Binary_search#Procedure
        //
        // Since runcount is stored as 16 bits, left + right
        // can't overflow.
        let left = 0;
        let right = this.runcount - 1;
        while (left <= right) {
            const mid = (left + right) >> 1;
            const i = mid * 4;
            const start = this.array[i] | (this.array[i + 1] << 8);
            const lenm1 = this.array[i + 2] | (this.array[i + 3] << 8);
            if ((start + lenm1) < value) {
                left = mid + 1;
            } else if (start > value) {
                right = mid - 1;
            } else {
                return true;
            }
        }
        return false;
    }
    * values() {
        let i = 0;
        while (i < this.runcount) {
            const start = this.array[i * 4] | (this.array[(i * 4) + 1] << 8);
            const lenm1 = this.array[(i * 4) + 2] | (this.array[(i * 4) + 3] << 8);
            let value = start;
            let j = 0;
            while (j <= lenm1) {
                yield value;
                value += 1;
                j += 1;
            }
            i += 1;
        }
    }
}
class RoaringBitmapArray {
    /**
     * @param {number} cardinality
     * @param {Uint8Array} array
     */
    constructor(cardinality, array) {
        this.cardinality = cardinality;
        this.array = array;
    }
    /** @param {number} value */
    contains(value) {
        // Binary search algorithm copied from
        // https://en.wikipedia.org/wiki/Binary_search#Procedure
        //
        // Since cardinality can't be higher than 4096, left + right
        // cannot overflow.
        let left = 0;
        let right = this.cardinality - 1;
        while (left <= right) {
            const mid = (left + right) >> 1;
            const i = mid * 2;
            const x = this.array[i] | (this.array[i + 1] << 8);
            if (x < value) {
                left = mid + 1;
            } else if (x > value) {
                right = mid - 1;
            } else {
                return true;
            }
        }
        return false;
    }
    /** @returns {Generator<number>} */
    * values() {
        let i = 0;
        const l = this.cardinality * 2;
        while (i < l) {
            yield this.array[i] | (this.array[i + 1] << 8);
            i += 2;
        }
    }
}
class RoaringBitmapBits {
    /**
     * @param {Uint8Array} array
     */
    constructor(array) {
        this.array = array;
    }
    /** @param {number} value */
    contains(value) {
        return !!(this.array[value >> 3] & (1 << (value & 7)));
    }
    * values() {
        let i = 0;
        const l = this.array.length << 3;
        while (i < l) {
            if (this.contains(i)) {
                yield i;
            }
            i += 1;
        }
    }
}

const EMPTY_BITMAP = new RoaringBitmap(null, 0);
EMPTY_BITMAP.consumed_len_bytes = 0;
const EMPTY_BITMAP1 = new RoaringBitmap(null, 0);
EMPTY_BITMAP1.consumed_len_bytes = 1;
const EVERYTHING_BITMAP = new RoaringBitmap(null, 0);

/**
 * A mapping from six byte nodeids to an arbitrary value.
 * We don't just use `Map` because that requires double hashing.
 * @template T
 * @property {Uint8Array} keys
 * @property {T[]} values
 * @property {number} size
 * @property {number} capacityClass
 */
class HashTable {
    /**
     * Construct an empty hash table.
     */
    constructor() {
        this.keys = EMPTY_UINT8;
        /** @type {(T|undefined)[]} */
        this.values = [];
        this.size = 0;
        this.capacityClass = 0;
    }
    /**
     * @returns {Generator<[Uint8Array, T]>}
     */
    * entries() {
        const keys = this.keys;
        const values = this.values;
        const l = this.values.length;
        for (let i = 0; i < l; i += 1) {
            const value = values[i];
            if (value !== undefined) {
                yield [keys.subarray(i * 6, (i + 1) * 6), value];
            }
        }
    }
    /**
     * Add a value to the hash table.
     * @param {Uint8Array} key
     * @param {T} value
     */
    set(key, value) {
        // 90 % load factor
        if (this.size * 10 >= this.values.length * 9) {
            const keys = this.keys;
            const values = this.values;
            const l = values.length;
            this.capacityClass += 1;
            const capacity = 1 << this.capacityClass;
            this.keys = new Uint8Array(capacity * 6);
            this.values = [];
            for (let i = 0; i < capacity; i += 1) {
                this.values.push(undefined);
            }
            this.size = 0;
            for (let i = 0; i < l; i += 1) {
                const oldValue = values[i];
                if (oldValue !== undefined) {
                    this.setNoGrow(keys, i * 6, oldValue);
                }
            }
        }
        this.setNoGrow(key, 0, value);
    }
    /**
     * @param {Uint8Array} key
     * @param {number} start
     * @param {T} value
     */
    setNoGrow(key, start, value) {
        const mask = ~(0xffffffff << this.capacityClass);
        const keys = this.keys;
        const values = this.values;
        const l = 1 << this.capacityClass;
        // because we know that our values are already hashed,
        // just chop off the lower four bytes
        let slot = (
            (key[start + 2] << 24) |
            (key[start + 3] << 16) |
            (key[start + 4] << 8) |
            key[start + 5]
        ) & mask;
        for (let distance = 0; distance < l; ) {
            const j = slot * 6;
            const otherValue = values[slot];
            if (otherValue === undefined) {
                values[slot] = value;
                const keysStart = slot * 6;
                keys[keysStart + 0] = key[start + 0];
                keys[keysStart + 1] = key[start + 1];
                keys[keysStart + 2] = key[start + 2];
                keys[keysStart + 3] = key[start + 3];
                keys[keysStart + 4] = key[start + 4];
                keys[keysStart + 5] = key[start + 5];
                this.size += 1;
                break;
            } else if (
                key[start + 0] === keys[j + 0] &&
                key[start + 1] === keys[j + 1] &&
                key[start + 2] === keys[j + 2] &&
                key[start + 3] === keys[j + 3] &&
                key[start + 4] === keys[j + 4] &&
                key[start + 5] === keys[j + 5]
            ) {
                values[slot] = value;
                break;
            } else {
                const otherPreferredSlot = (
                    (keys[j + 2] << 24) | (keys[j + 3] << 16) |
                    (keys[j + 4] << 8) | keys[j + 5]
                ) & mask;
                const otherDistance = otherPreferredSlot <= slot ?
                    slot - otherPreferredSlot :
                    (l - otherPreferredSlot) + slot;
                if (distance > otherDistance) {
                    // if the other key is closer to its preferred slot than this one,
                    // then insert our node in its place and swap
                    //
                    // https://cglab.ca/~abeinges/blah/robinhood-part-1/
                    const otherKey = keys.slice(j, j + 6);
                    values[slot] = value;
                    value = otherValue;
                    keys[j + 0] = key[start + 0];
                    keys[j + 1] = key[start + 1];
                    keys[j + 2] = key[start + 2];
                    keys[j + 3] = key[start + 3];
                    keys[j + 4] = key[start + 4];
                    keys[j + 5] = key[start + 5];
                    key = otherKey;
                    start = 0;
                    distance = otherDistance;
                }
                distance += 1;
                slot = (slot + 1) & mask;
            }
        }
    }
    /**
     * Retrieve a value
     * @param {Uint8Array} key
     * @returns {T|undefined}
     */
    get(key) {
        if (key.length !== 6) {
            throw "invalid key";
        }
        return this.getWithOffsetKey(key, 0);
    }
    /**
     * Retrieve a value
     * @param {Uint8Array} key
     * @param {number} start
     * @returns {T|undefined}
     */
    getWithOffsetKey(key, start) {
        const mask = ~(0xffffffff << this.capacityClass);
        const keys = this.keys;
        const values = this.values;
        const l = 1 << this.capacityClass;
        // because we know that our values are already hashed,
        // just chop off the lower four bytes
        let slot = (
            (key[start + 2] << 24) |
            (key[start + 3] << 16) |
            (key[start + 4] << 8) |
            key[start + 5]
        ) & mask;
        for (let distance = 0; distance < l; distance += 1) {
            const j = slot * 6;
            const value = values[slot];
            if (value === undefined) {
                break;
            } else if (
                key[start + 0] === keys[j + 0] &&
                key[start + 1] === keys[j + 1] &&
                key[start + 2] === keys[j + 2] &&
                key[start + 3] === keys[j + 3] &&
                key[start + 4] === keys[j + 4] &&
                key[start + 5] === keys[j + 5]
            ) {
                return value;
            } else {
                const otherPreferredSlot = (
                    (keys[j + 2] << 24) | (keys[j + 3] << 16) |
                    (keys[j + 4] << 8) | keys[j + 5]
                ) & mask;
                const otherDistance = otherPreferredSlot <= slot ?
                    slot - otherPreferredSlot :
                    (l - otherPreferredSlot) + slot;
                if (distance > otherDistance) {
                    break;
                }
            }
            slot = (slot + 1) & mask;
        }
        return undefined;
    }
}

/*eslint-disable */
// ignore-tidy-linelength
/** <https://stackoverflow.com/questions/43122082/efficiently-count-the-number-of-bits-in-an-integer-in-javascript>
 * @param {number} n
 * @returns {number}
 */
function bitCount(n) {
    n = (~~n) - ((n >> 1) & 0x55555555);
    n = (n & 0x33333333) + ((n >> 2) & 0x33333333);
    return ((n + (n >> 4) & 0xF0F0F0F) * 0x1010101) >> 24;
}
/*eslint-enable */

/**
 * https://en.wikipedia.org/wiki/Boyer%E2%80%93Moore%E2%80%93Horspool_algorithm
 */
class Uint8ArraySearchPattern {
    /** @param {Uint8Array} needle */
    constructor(needle) {
        this.needle = needle;
        this.skipTable = [];
        const m = needle.length;
        for (let i = 0; i < 256; i += 1) {
            this.skipTable.push(m);
        }
        for (let i = 0; i < m - 1; i += 1) {
            this.skipTable[needle[i]] = m - 1 - i;
        }
    }
    /**
     * @param {Uint8Array} haystack
     * @returns {boolean}
     */
    matches(haystack) {
        const needle = this.needle;
        const skipTable = this.skipTable;
        const m = needle.length;
        const n = haystack.length;

        let skip = 0;
        search: while (n - skip >= m) {
            for (let i = m - 1; i >= 0; i -= 1) {
                if (haystack[skip + i] !== needle[i]) {
                    skip += skipTable[haystack[skip + m  - 1]];
                    continue search;
                }
            }
            return true;
        }
        return false;
    }
}

/**
 * @param {stringdex.Hooks} hooks
 * @returns {Promise<stringdex.Database>}
 */
function loadDatabase(hooks) {
    /** @type {stringdex.Callbacks} */
    const callbacks = {
        rr_: function(data) {
            const dataObj = JSON.parse(data);
            for (const colName of Object.keys(dataObj)) {
                if (Object.hasOwn(dataObj[colName], "N")) {
                    const counts = [];
                    const countsstring = dataObj[colName]["N"];
                    let i = 0;
                    const l = countsstring.length;
                    while (i < l) {
                        let n = 0;
                        let c = countsstring.charCodeAt(i);
                        while (c < 96) { // 96 = "`"
                            n = (n << 4) | (c & 0xF);
                            i += 1;
                            c = countsstring.charCodeAt(i);
                        }
                        n = (n << 4) | (c & 0xF);
                        counts.push(n);
                        i += 1;
                    }
                    registry.dataColumns.set(colName, new DataColumn(
                        counts,
                        makeUint8ArrayFromBase64(dataObj[colName]["H"]),
                        new RoaringBitmap(makeUint8ArrayFromBase64(dataObj[colName]["E"]), 0),
                        colName,
                        Object.hasOwn(dataObj[colName], "I") ?
                            makeSearchTreeFromBase64(dataObj[colName].I)[1] :
                            null,
                    ));
                }
            }
            const cb = registry.searchTreeRootCallback;
            if (cb) {
                cb(null, new Database(registry.searchTreeRoots, registry.dataColumns));
            }
        },
        err_rr_: function(err) {
            const cb = registry.searchTreeRootCallback;
            if (cb) {
                cb(err, null);
            }
        },
        rd_: function(dataString) {
            const l = dataString.length;
            const data = new Uint8Array(l);
            for (let i = 0; i < l; ++i) {
                data[i] = dataString.charCodeAt(i);
            }
            loadColumnFromBytes(data);
        },
        err_rd_: function(filename, err) {
            const nodeid = makeUint8ArrayFromHex(filename);
            const cb = registry.dataColumnLoadPromiseCallbacks.get(nodeid);
            if (cb) {
                cb(err, null);
            }
        },
        rb_: function(dataString64) {
            loadColumnFromBytes(makeUint8ArrayFromBase64(dataString64));
        },
        err_rb_: function(filename, err) {
            const nodeid = makeUint8ArrayFromHex(filename);
            const cb = registry.dataColumnLoadPromiseCallbacks.get(nodeid);
            if (cb) {
                cb(err, null);
            }
        },
        rn_: function(inputBase64) {
            const [nodeid, tree] = makeSearchTreeFromBase64(inputBase64);
            const cb = registry.searchTreeLoadPromiseCallbacks.get(nodeid);
            if (cb) {
                cb(null, tree);
                registry.searchTreeLoadPromiseCallbacks.set(nodeid, null);
            }
        },
        err_rn_: function(filename, err) {
            const nodeid = makeUint8ArrayFromHex(filename);
            const cb = registry.searchTreeLoadPromiseCallbacks.get(nodeid);
            if (cb) {
                cb(err, null);
            }
        },
    };

    /**
     * @type {{
     *      searchTreeRoots: Map<string, SearchTree>;
     *      searchTreeLoadPromiseCallbacks: HashTable<(function(any, SearchTree?): any)|null>;
     *      searchTreePromises: HashTable<Promise<SearchTree>>;
     *      dataColumnLoadPromiseCallbacks: HashTable<function(any, Uint8Array[]?): any>;
     *      dataColumns: Map<string, DataColumn>;
     *      dataColumnsBuckets: HashTable<Promise<Uint8Array[]>>;
     *      searchTreeLoadByNodeID: function(Uint8Array): Promise<SearchTree>;
     *      searchTreeRootCallback?: function(any, Database?): any;
     *      dataLoadByNameAndHash: function(string, Uint8Array): Promise<Uint8Array[]>;
     * }}
     */
    const registry = {
        searchTreeRoots: new Map(),
        searchTreeLoadPromiseCallbacks: new HashTable(),
        searchTreePromises: new HashTable(),
        dataColumnLoadPromiseCallbacks: new HashTable(),
        dataColumns: new Map(),
        dataColumnsBuckets: new HashTable(),
        searchTreeLoadByNodeID: function(nodeid) {
            const existingPromise = registry.searchTreePromises.get(nodeid);
            if (existingPromise) {
                return existingPromise;
            }
            /** @type {Promise<SearchTree>} */
            let newPromise;
            if ((nodeid[0] & 0x80) !== 0) {
                const isWhole = (nodeid[0] & 0x40) !== 0;
                let leaves;
                if ((nodeid[0] & 0x10) !== 0) {
                    let id1 = (nodeid[2] << 8) | nodeid[3];
                    if ((nodeid[0] & 0x20) !== 0) {
                        // when data is present, id1 can be up to 20 bits
                        id1 |= ((nodeid[1] & 0x0f) << 16);
                    } else {
                        // otherwise, we fit in 28
                        id1 |= ((nodeid[0] & 0x0f) << 24) | (nodeid[1] << 16);
                    }
                    const id2 = id1 + ((nodeid[4] << 8) | nodeid[5]);
                    leaves = RoaringBitmap.makeSingleton(id1)
                        .union(RoaringBitmap.makeSingleton(id2));
                } else {
                    leaves = RoaringBitmap.makeSingleton(
                        (nodeid[2] << 24) | (nodeid[3] << 16) |
                        (nodeid[4] << 8) | nodeid[5],
                    );
                }
                const data = (nodeid[0] & 0x20) !== 0 ?
                    Uint8Array.of(((nodeid[0] & 0x0f) << 4) | (nodeid[1] >> 4)) :
                    EMPTY_UINT8;
                newPromise = Promise.resolve(new PrefixSearchTree(
                    EMPTY_SEARCH_TREE_BRANCHES,
                    EMPTY_SEARCH_TREE_BRANCHES,
                    data,
                    isWhole ? leaves : EMPTY_BITMAP,
                    isWhole ? EMPTY_BITMAP : leaves,
                ));
            } else {
                const hashHex = makeHexFromUint8Array(nodeid);
                newPromise = new Promise((resolve, reject) => {
                    const cb = registry.searchTreeLoadPromiseCallbacks.get(nodeid);
                    if (cb) {
                        registry.searchTreeLoadPromiseCallbacks.set(nodeid, (err, data) => {
                            cb(err, data);
                            if (data) {
                                resolve(data);
                            } else {
                                reject(err);
                            }
                        });
                    } else {
                        registry.searchTreeLoadPromiseCallbacks.set(nodeid, (err, data) => {
                            if (data) {
                                resolve(data);
                            } else {
                                reject(err);
                            }
                        });
                        hooks.loadTreeByHash(hashHex);
                    }
                });
            }
            registry.searchTreePromises.set(nodeid, newPromise);
            return newPromise;
        },
        dataLoadByNameAndHash: function(name, hash) {
            const existingBucket = registry.dataColumnsBuckets.get(hash);
            if (existingBucket) {
                return existingBucket;
            }
            const hashHex = makeHexFromUint8Array(hash);
            /** @type {Promise<Uint8Array[]>} */
            const newBucket = new Promise((resolve, reject) => {
                const cb = registry.dataColumnLoadPromiseCallbacks.get(hash);
                if (cb) {
                    registry.dataColumnLoadPromiseCallbacks.set(hash, (err, data) => {
                        cb(err, data);
                        if (data) {
                            resolve(data);
                        } else {
                            reject(err);
                        }
                    });
                } else {
                    registry.dataColumnLoadPromiseCallbacks.set(hash, (err, data) => {
                        if (data) {
                            resolve(data);
                        } else {
                            reject(err);
                        }
                    });
                    hooks.loadDataByNameAndHash(name, hashHex);
                }
            });
            registry.dataColumnsBuckets.set(hash, newBucket);
            return newBucket;
        },
    };

    /**
     * The set of child subtrees.
     * @template ST
     * @type {{
     *    nodeids: Uint8Array,
     *    subtrees: Array<Promise<ST>|null>,
     * }}
     */
    class SearchTreeBranches {
        /**
         * Construct the subtree list with `length` nulls
         * @param {number} length
         * @param {Uint8Array} nodeids
         */
        constructor(length, nodeids) {
            this.nodeids = nodeids;
            this.subtrees = [];
            for (let i = 0; i < length; ++i) {
                this.subtrees.push(null);
            }
        }
        /**
         * @param {number} i
         * @returns {Uint8Array}
        */
        getNodeID(i) {
            return new Uint8Array(
                this.nodeids.buffer,
                this.nodeids.byteOffset + (i * 6),
                6,
            );
        }
        // https://github.com/microsoft/TypeScript/issues/17227
        /** @returns {Generator<[number, Promise<ST>|null]>} */
        entries() {
            throw new Error();
        }
        /**
         * @param {number} _k
         * @returns {number}
         */
        getIndex(_k) {
            throw new Error();
        }
        /**
         * @param {number} _i
         * @returns {number}
         */
        getKey(_i) {
            throw new Error();
        }
        /**
         * @returns {Uint8Array}
         */
        getKeys() {
            throw new Error();
        }
    }

    /**
     * A sorted array of search tree branches.
     *
     * @template ST
     * @extends SearchTreeBranches<ST>
     * @type {{
     *    keys: Uint8Array,
     *    nodeids: Uint8Array,
     *    subtrees: Array<Promise<ST>|null>,
     * }}
     */
    class SearchTreeBranchesArray extends SearchTreeBranches {
        /**
         * @param {Uint8Array} keys
         * @param {Uint8Array} nodeids
         */
        constructor(keys, nodeids) {
            super(keys.length, nodeids);
            this.keys = keys;
            let i = 1;
            while (i < this.keys.length) {
                if (this.keys[i - 1] >= this.keys[i]) {
                    throw new Error("HERE");
                }
                i += 1;
            }
        }
        /** @returns {Generator<[number, Promise<ST>|null]>} */
        * entries() {
            let i = 0;
            const l = this.keys.length;
            while (i < l) {
                yield [this.keys[i], this.subtrees[i]];
                i += 1;
            }
        }
        /**
         * @param {number} k
         * @returns {number}
         */
        getIndex(k) {
            // Since length can't be bigger than 256,
            // left + right can't overflow.
            let left = 0;
            let right = this.keys.length - 1;
            while (left <= right) {
                const mid = (left + right) >> 1;
                if (this.keys[mid] < k) {
                    left = mid + 1;
                } else if (this.keys[mid] > k) {
                    right = mid - 1;
                } else {
                    return mid;
                }
            }
            return -1;
        }
        /**
         * @param {number} i
         * @returns {number}
         */
        getKey(i) {
            return this.keys[i];
        }
        /**
         * @returns {Uint8Array}
         */
        getKeys() {
            return this.keys;
        }
    }

    const EMPTY_SEARCH_TREE_BRANCHES = new SearchTreeBranchesArray(
        EMPTY_UINT8,
        EMPTY_UINT8,
    );

    /** @type {number[]} */
    const SHORT_ALPHABITMAP_CHARS = [];
    for (let i = 0x61; i <= 0x7A; ++i) {
        if (i === 0x76 || i === 0x71) {
            // 24 entries, 26 letters, so we skip q and v
            continue;
        }
        SHORT_ALPHABITMAP_CHARS.push(i);
    }

    /** @type {number[]} */
    const LONG_ALPHABITMAP_CHARS = [0x31, 0x32, 0x33, 0x34, 0x35, 0x36];
    for (let i = 0x61; i <= 0x7A; ++i) {
        LONG_ALPHABITMAP_CHARS.push(i);
    }

    /**
     * @template ST
     * @param {number[]} alphabitmap_chars
     * @param {number} width
     * @return {(typeof SearchTreeBranches<ST>)&{"ALPHABITMAP_CHARS": number[], "width": number}}
     */
    function makeSearchTreeBranchesAlphaBitmapClass(alphabitmap_chars, width) {
        const bitwidth = width * 8;
        /**
         * @extends SearchTreeBranches<ST>
         */
        const cls = class SearchTreeBranchesAlphaBitmap extends SearchTreeBranches {
            /**
             * @param {number} bitmap
             * @param {Uint8Array} nodeids
             */
            constructor(bitmap, nodeids) {
                super(nodeids.length / 6, nodeids);
                if (nodeids.length / 6 !== bitCount(bitmap)) {
                    throw new Error(`mismatch ${bitmap} ${nodeids}`);
                }
                this.bitmap = bitmap;
                this.nodeids = nodeids;
            }
            /** @returns {Generator<[number, Promise<ST>|null]>} */
            * entries() {
                let i = 0;
                let j = 0;
                while (i < bitwidth) {
                    if (this.bitmap & (1 << i)) {
                        yield [alphabitmap_chars[i], this.subtrees[j]];
                        j += 1;
                    }
                    i += 1;
                }
            }
            /**
             * @param {number} k
             * @returns {number}
             */
            getIndex(k) {
                //return this.getKeys().indexOf(k);
                const ix = alphabitmap_chars.indexOf(k);
                if (ix < 0) {
                    return ix;
                }
                const result = bitCount(~(0xffffffff << ix) & this.bitmap);
                return result >= this.subtrees.length ? -1 : result;
            }
            /**
             * @param {number} branch_index
             * @returns {number}
             */
            getKey(branch_index) {
                return this.getKeys()[branch_index];
            }
            /**
             * @returns {Uint8Array}
             */
            getKeys() {
                const length = bitCount(this.bitmap);
                const result = new Uint8Array(length);
                let result_index = 0;
                for (let alpha_index = 0; alpha_index < bitwidth; ++alpha_index) {
                    if (this.bitmap & (1 << alpha_index)) {
                        result[result_index] = alphabitmap_chars[alpha_index];
                        result_index += 1;
                    }
                }
                return result;
            }
        };
        cls.ALPHABITMAP_CHARS = alphabitmap_chars;
        cls.width = width;
        return cls;
    }

    /**
     * @template ST
     * @type {(typeof SearchTreeBranches<any>)&{"ALPHABITMAP_CHARS": number[], "width": number}}
     */
    const SearchTreeBranchesShortAlphaBitmap =
        makeSearchTreeBranchesAlphaBitmapClass(SHORT_ALPHABITMAP_CHARS, 3);

    /**
     * @template ST
     * @type {(typeof SearchTreeBranches<any>)&{"ALPHABITMAP_CHARS": number[], "width": number}}
     */
    const SearchTreeBranchesLongAlphaBitmap =
        makeSearchTreeBranchesAlphaBitmapClass(LONG_ALPHABITMAP_CHARS, 4);

    /**
     * @typedef {PrefixSearchTree|SuffixSearchTree} SearchTree
     * @typedef {PrefixTrie|SuffixTrie} Trie
     */

    /**
     * An interleaved [prefix] and [suffix tree],
     * used for name-based search.
     *
     * This data structure is used to drive prefix matches,
     * such as matching the query "link" to `LinkedList`,
     * and Lev-distance matches, such as matching the
     * query "hahsmap" to `HashMap`.
     *
     * [prefix tree]: https://en.wikipedia.org/wiki/Prefix_tree
     * [suffix tree]: https://en.wikipedia.org/wiki/Suffix_tree
     *
     * branches
     * : A sorted-array map of subtrees.
     *
     * data
     * : The substring represented by this node. The root node
     *   is always empty.
     *
     * leaves_suffix
     * : The IDs of every entry that matches. Levenshtein matches
     *   won't include these.
     *
     * leaves_whole
     * : The IDs of every entry that matches exactly. Levenshtein matches
     *   will include these.
     *
     * @type {{
     *     might_have_prefix_branches: SearchTreeBranches<SearchTree>,
     *     branches: SearchTreeBranches<SearchTree>,
     *     data: Uint8Array,
     *     leaves_suffix: RoaringBitmap,
     *     leaves_whole: RoaringBitmap,
     * }}
     */
    class PrefixSearchTree {
        /**
         * @param {SearchTreeBranches<SearchTree>} branches
         * @param {SearchTreeBranches<SearchTree>} might_have_prefix_branches
         * @param {Uint8Array} data
         * @param {RoaringBitmap} leaves_whole
         * @param {RoaringBitmap} leaves_suffix
         */
        constructor(
            branches,
            might_have_prefix_branches,
            data,
            leaves_whole,
            leaves_suffix,
        ) {
            this.might_have_prefix_branches = might_have_prefix_branches;
            this.branches = branches;
            this.data = data;
            this.leaves_suffix = leaves_suffix;
            this.leaves_whole = leaves_whole;
        }
        /**
         * Returns the Trie for the root node.
         *
         * A Trie pointer refers to a single node in a logical decompressed search tree
         * (the real search tree is compressed).
         *
         * @param {DataColumn} dataColumn
         * @param {Uint8ArraySearchPattern} searchPattern
         * @return {PrefixTrie}
         */
        trie(dataColumn, searchPattern) {
            return new PrefixTrie(this, 0, dataColumn, searchPattern);
        }

        /**
         * Return the trie representing `name`
         * @param {Uint8Array|string} name
         * @param {DataColumn} dataColumn
         * @returns {Promise<Trie?>}
         */
        async search(name, dataColumn) {
            if (typeof name === "string") {
                const utf8encoder = new TextEncoder();
                name = utf8encoder.encode(name);
            }
            const searchPattern = new Uint8ArraySearchPattern(name);
            /** @type {Trie} */
            let trie = this.trie(dataColumn, searchPattern);
            for (const datum of name) {
                // code point definitely exists
                /** @type {Promise<Trie>?} */
                const newTrie = trie.child(datum);
                if (newTrie) {
                    trie = await newTrie;
                } else {
                    return null;
                }
            }
            return trie;
        }

        /**
         * @param {Uint8Array|string} name
         * @param {DataColumn} dataColumn
         * @returns {AsyncGenerator<Trie>}
         */
        async* searchLev(name, dataColumn) {
            if (typeof name === "string") {
                const utf8encoder = new TextEncoder();
                name = utf8encoder.encode(name);
            }
            const w = name.length;
            if (w < 3) {
                const trie = await this.search(name, dataColumn);
                if (trie !== null) {
                    yield trie;
                }
                return;
            }
            const searchPattern = new Uint8ArraySearchPattern(name);
            const levParams = w >= 6 ?
                new Lev2TParametricDescription(w) :
                new Lev1TParametricDescription(w);
            /** @type {Array<[Promise<Trie>, number]>} */
            const stack = [[Promise.resolve(this.trie(dataColumn, searchPattern)), 0]];
            const n = levParams.n;
            while (stack.length !== 0) {
                // It's not empty
                /** @type {[Promise<Trie>, number]} */
                //@ts-expect-error
                const [triePromise, levState] = stack.pop();
                const trie = await triePromise;
                for (const byte of trie.keysExcludeSuffixOnly()) {
                    const levPos = levParams.getPosition(levState);
                    const vector = levParams.getVector(
                        name,
                        byte,
                        levPos,
                        Math.min(w, levPos + (2 * n) + 1),
                    );
                    const newLevState = levParams.transition(
                        levState,
                        levPos,
                        vector,
                    );
                    if (newLevState >= 0) {
                        const child = trie.child(byte);
                        if (child) {
                            stack.push([child, newLevState]);
                            if (levParams.isAccept(newLevState)) {
                                yield child;
                            }
                        }
                    }
                }
            }
        }

        /** @returns {RoaringBitmap} */
        getCurrentLeaves() {
            return this.leaves_whole.union(this.leaves_suffix);
        }
    }

    /**
     * A representation of a set of strings in the search index,
     * as a subset of the entire tree.
     */
    class PrefixTrie {
        /**
         * @param {PrefixSearchTree} tree
         * @param {number} offset
         * @param {DataColumn} dataColumn
         * @param {Uint8ArraySearchPattern} searchPattern
         */
        constructor(tree, offset, dataColumn, searchPattern) {
            this.tree = tree;
            this.offset = offset;
            this.dataColumn = dataColumn;
            this.searchPattern = searchPattern;
        }

        /**
         * All exact matches for the string represented by this node.
         * @returns {RoaringBitmap}
         */
        matches() {
            if (this.offset === this.tree.data.length) {
                return this.tree.leaves_whole;
            } else {
                return EMPTY_BITMAP;
            }
        }

        /**
         * All matches for strings that contain the string represented by this node.
         * @returns {AsyncGenerator<RoaringBitmap>}
         */
        async* substringMatches() {
            /** @type {Promise<SearchTree>[]} */
            let layer = [Promise.resolve(this.tree)];
            while (layer.length) {
                const current_layer = layer;
                layer = [];
                for await (const tree of current_layer) {
                    /** @type {number[]?} */
                    let rejected = null;
                    let leaves = tree.getCurrentLeaves();
                    for (const leaf of leaves.entries()) {
                        const haystack = await this.dataColumn.at(leaf);
                        if (haystack === undefined || !this.searchPattern.matches(haystack)) {
                            if (!rejected) {
                                rejected = [];
                            }
                            rejected.push(leaf);
                        }
                    }
                    if (rejected) {
                        if (leaves.cardinality() !== rejected.length) {
                            for (const rej of rejected) {
                                leaves = leaves.remove(rej);
                            }
                            yield leaves;
                        }
                    } else {
                        yield leaves;
                    }
                }
                /** @type {HashTable<[number, SearchTree][]>} */
                const subnodes = new HashTable();
                for await (const node of current_layer) {
                    const branches = node.branches;
                    const l = branches.subtrees.length;
                    for (let i = 0; i < l; ++i) {
                        const subtree = branches.subtrees[i];
                        if (subtree) {
                            layer.push(subtree);
                        } else if (subtree === null) {
                            const byte = branches.getKey(i);
                            const newnode = branches.getNodeID(i);
                            if (!newnode) {
                                throw new Error(`malformed tree; no node for key ${byte}`);
                            } else {
                                let subnode_list = subnodes.get(newnode);
                                if (!subnode_list) {
                                    subnode_list = [[byte, node]];
                                    subnodes.set(newnode, subnode_list);
                                } else {
                                    subnode_list.push([byte, node]);
                                }
                            }
                        } else {
                            throw new Error(`malformed tree; index ${i} does not exist`);
                        }
                    }
                }
                for (const [newnode, subnode_list] of subnodes.entries()) {
                    const res = registry.searchTreeLoadByNodeID(newnode);
                    for (const [byte, node] of subnode_list) {
                        const branches = node.branches;
                        const i = branches.getIndex(byte);
                        branches.subtrees[i] = res;
                        if (node instanceof PrefixSearchTree) {
                            const might_have_prefix_branches = node.might_have_prefix_branches;
                            const mhpI = might_have_prefix_branches.getIndex(byte);
                            if (mhpI !== -1) {
                                might_have_prefix_branches.subtrees[mhpI] = res;
                            }
                        }
                    }
                    layer.push(res);
                }
            }
        }

        /**
         * All matches for strings that start with the string represented by this node.
         * @returns {AsyncGenerator<RoaringBitmap>}
         */
        async* prefixMatches() {
            /** @type {{node: Promise<SearchTree>, len: number}[]} */
            let layer = [{node: Promise.resolve(this.tree), len: 0}];
            // https://en.wikipedia.org/wiki/Heap_(data_structure)#Implementation_using_arrays
            /** @type {{bitmap: RoaringBitmap, length: number}[]} */
            const backlog = [];
            while (layer.length !== 0 || backlog.length !== 0) {
                const current_layer = layer;
                layer = [];
                let minLength = null;
                // push every entry in the current layer into the backlog,
                // a min-heap of result entries
                // we then yield the smallest ones (can't yield bigger ones
                // if we want to do them in order)
                for (const {node, len} of current_layer) {
                    const tree = await node;
                    if (!(tree instanceof PrefixSearchTree)) {
                        continue;
                    }
                    const length = len + tree.data.length;
                    if (minLength === null || length < minLength) {
                        minLength = length;
                    }
                    let backlogSlot = backlog.length;
                    backlog.push({bitmap: tree.leaves_whole, length});
                    while (backlogSlot > 0 &&
                        backlog[backlogSlot].length < backlog[(backlogSlot - 1) >> 1].length
                    ) {
                        const parentSlot = (backlogSlot - 1) >> 1;
                        const parent = backlog[parentSlot];
                        backlog[parentSlot] = backlog[backlogSlot];
                        backlog[backlogSlot] = parent;
                        backlogSlot = parentSlot;
                    }
                }
                // yield nodes in length order, smallest one first
                // we know that, whatever the smallest item is
                // every child will be bigger than that
                while (backlog.length !== 0) {
                    const backlogEntry = backlog[0];
                    if (minLength !== null && backlogEntry.length > minLength) {
                        break;
                    }
                    if (!backlogEntry.bitmap.isEmpty()) {
                        yield backlogEntry.bitmap;
                    }
                    backlog[0] = backlog[backlog.length - 1];
                    backlog.length -= 1;
                    let backlogSlot = 0;
                    const backlogLength = backlog.length;
                    while (backlogSlot < backlogLength) {
                        const leftSlot = (backlogSlot << 1) + 1;
                        const rightSlot = (backlogSlot << 1) + 2;
                        let smallest = backlogSlot;
                        if (leftSlot < backlogLength &&
                            backlog[leftSlot].length < backlog[smallest].length
                        ) {
                            smallest = leftSlot;
                        }
                        if (rightSlot < backlogLength &&
                            backlog[rightSlot].length < backlog[smallest].length
                        ) {
                            smallest = rightSlot;
                        }
                        if (smallest === backlogSlot) {
                            break;
                        } else {
                            const tmp = backlog[backlogSlot];
                            backlog[backlogSlot] = backlog[smallest];
                            backlog[smallest] = tmp;
                            backlogSlot = smallest;
                        }
                    }
                }
                // if we still have more subtrees to walk, then keep going
                /** @type {HashTable<{byte: number, tree: PrefixSearchTree, len: number}[]>} */
                const subnodes = new HashTable();
                for await (const {node, len} of current_layer) {
                    const tree = await node;
                    if (!(tree instanceof PrefixSearchTree)) {
                        continue;
                    }
                    const length = len + tree.data.length;
                    const mhp_branches = tree.might_have_prefix_branches;
                    const l = mhp_branches.subtrees.length;
                    for (let i = 0; i < l; ++i) {
                        const len = length + 1;
                        const subtree = mhp_branches.subtrees[i];
                        if (subtree) {
                            layer.push({node: subtree, len});
                        } else if (subtree === null) {
                            const byte = mhp_branches.getKey(i);
                            const newnode = mhp_branches.getNodeID(i);
                            if (!newnode) {
                                throw new Error(`malformed tree; no node for key ${byte}`);
                            } else {
                                let subnode_list = subnodes.get(newnode);
                                if (!subnode_list) {
                                    subnode_list = [{byte, tree, len}];
                                    subnodes.set(newnode, subnode_list);
                                } else {
                                    subnode_list.push({byte, tree, len});
                                }
                            }
                        }
                    }
                }
                for (const [newnode, subnode_list] of subnodes.entries()) {
                    const res = registry.searchTreeLoadByNodeID(newnode);
                    let len = Number.MAX_SAFE_INTEGER;
                    for (const {byte, tree, len: subtreelen} of subnode_list) {
                        if (subtreelen < len) {
                            len = subtreelen;
                        }
                        const mhp_branches = tree.might_have_prefix_branches;
                        const i = mhp_branches.getIndex(byte);
                        mhp_branches.subtrees[i] = res;
                        const branches = tree.branches;
                        const bi = branches.getIndex(byte);
                        branches.subtrees[bi] = res;
                    }
                    layer.push({node: res, len});
                }
            }
        }

        /**
         * Returns all keys that are children of this node.
         * @returns {Uint8Array}
         */
        keys() {
            const data = this.tree.data;
            if (this.offset === data.length) {
                return this.tree.branches.getKeys();
            } else {
                return Uint8Array.of(data[this.offset]);
            }
        }

        /**
         * Returns all nodes that are direct children of this node.
         * @returns {[number, Promise<Trie>][]}
         */
        children() {
            const data = this.tree.data;
            if (this.offset === data.length) {
                /** @type {[number, Promise<Trie>][]} */
                const nodes = [];
                let i = 0;
                for (const [k, v] of this.tree.branches.entries()) {
                    /** @type {Promise<SearchTree>} */
                    let node;
                    if (v) {
                        node = v;
                    } else {
                        const newnode = this.tree.branches.getNodeID(i);
                        if (!newnode) {
                            throw new Error(`malformed tree; no hash for key ${k}: ${newnode} \
                                ${this.tree.branches.nodeids} ${this.tree.branches.getKeys()}`);
                        }
                        node = registry.searchTreeLoadByNodeID(newnode);
                        this.tree.branches.subtrees[i] = node;
                        const mhpI = this.tree.might_have_prefix_branches.getIndex(k);
                        if (mhpI !== -1) {
                            this.tree.might_have_prefix_branches.subtrees[mhpI] = node;
                        }
                    }
                    nodes.push([k, node.then(node => {
                        return node.trie(this.dataColumn, this.searchPattern);
                    })]);
                    i += 1;
                }
                return nodes;
            } else {
                /** @type {number} */
                const codePoint = data[this.offset];
                const trie = new PrefixTrie(
                    this.tree,
                    this.offset + 1,
                    this.dataColumn,
                    this.searchPattern,
                );
                return [[codePoint, Promise.resolve(trie)]];
            }
        }

        /**
         * Returns all keys that are children of this node.
         * @returns {Uint8Array}
         */
        keysExcludeSuffixOnly() {
            const data = this.tree.data;
            if (this.offset === data.length) {
                return this.tree.might_have_prefix_branches.getKeys();
            } else {
                return Uint8Array.of(data[this.offset]);
            }
        }

        /**
         * Returns all nodes that are direct children of this node.
         * @returns {[number, Promise<Trie>][]}
         */
        childrenExcludeSuffixOnly() {
            const data = this.tree.data;
            if (this.offset === data.length) {
                /** @type {[number, Promise<Trie>][]} */
                const nodes = [];
                let i = 0;
                for (const [k, v] of this.tree.might_have_prefix_branches.entries()) {
                    /** @type {Promise<SearchTree>} */
                    let node;
                    if (v) {
                        node = v;
                    } else {
                        const newnode = this.tree.might_have_prefix_branches.getNodeID(i);
                        if (!newnode) {
                            throw new Error(`malformed tree; no node for key ${k}`);
                        }
                        node = registry.searchTreeLoadByNodeID(newnode);
                        this.tree.might_have_prefix_branches.subtrees[i] = node;
                        this.tree.branches.subtrees[this.tree.branches.getIndex(k)] = node;
                    }
                    nodes.push([k, node.then(node => {
                        return node.trie(this.dataColumn, this.searchPattern);
                    })]);
                    i += 1;
                }
                return nodes;
            } else {
                /** @type {number} */
                const codePoint = data[this.offset];
                const trie = new PrefixTrie(
                    this.tree,
                    this.offset + 1,
                    this.dataColumn,
                    this.searchPattern,
                );
                return [[codePoint, Promise.resolve(trie)]];
            }
        }

        /**
         * Returns a single node that is a direct child of this node.
         * @param {number} byte
         * @returns {Promise<Trie>?}
         */
        child(byte) {
            if (this.offset === this.tree.data.length) {
                const i = this.tree.branches.getIndex(byte);
                if (i !== -1) {
                    let branch = this.tree.branches.subtrees[i];
                    if (branch === null) {
                        const newnode = this.tree.branches.getNodeID(i);
                        if (!newnode) {
                            throw new Error(`malformed tree; no node for key ${byte}`);
                        }
                        branch = registry.searchTreeLoadByNodeID(newnode);
                        this.tree.branches.subtrees[i] = branch;
                        const mhpI = this.tree.might_have_prefix_branches.getIndex(byte);
                        if (mhpI !== -1) {
                            this.tree.might_have_prefix_branches.subtrees[mhpI] = branch;
                        }
                    }
                    return branch.then(branch => branch.trie(this.dataColumn, this.searchPattern));
                }
            } else if (this.tree.data[this.offset] === byte) {
                return Promise.resolve(new PrefixTrie(
                    this.tree,
                    this.offset + 1,
                    this.dataColumn,
                    this.searchPattern,
                ));
            }
            return null;
        }
    }
    /**
     * A [suffix tree], used for name-based search.
     *
     * This data structure is used to drive substring matches,
     * such as matching the query "inked" to `LinkedList`.
     *
     * [suffix tree]: https://en.wikipedia.org/wiki/Suffix_tree
     *
     * Suffix trees do not actually carry the intermediate data
     * between branches, so, in order to validate the results,
     * they must go through a filtering step at the end.
     * Suffix trees also cannot match lev and exact matches,
     * so those just return empty sets.
     *
     * branches
     * : A sorted-array map of subtrees.
     *
     * dataLen
     * : The length of the substring used by this node.
     *
     * leaves_suffix
     * : The IDs of every entry that matches. Levenshtein matches
     *   won't include these.
     *
     * @type {{
     *     branches: SearchTreeBranches<SearchTree>,
     *     dataLen: number,
     *     leaves_suffix: RoaringBitmap,
     * }}
     */
    class SuffixSearchTree {
        /**
         * @param {SearchTreeBranches<SearchTree>} branches
         * @param {number} dataLen
         * @param {RoaringBitmap} leaves_suffix
         */
        constructor(
            branches,
            dataLen,
            leaves_suffix,
        ) {
            this.branches = branches;
            this.dataLen = dataLen;
            this.leaves_suffix = leaves_suffix;
        }
        /**
         * Returns the Trie for the root node.
         *
         * A Trie pointer refers to a single node in a logical decompressed search tree
         * (the real search tree is compressed).
         *
         * @param {DataColumn} dataColumn
         * @param {Uint8ArraySearchPattern} searchPattern
         * @return {Trie}
         */
        trie(dataColumn, searchPattern) {
            return new SuffixTrie(this, 0, dataColumn, searchPattern);
        }

        /**
         * Return the trie representing `name`
         * @param {Uint8Array|string} name
         * @param {DataColumn} dataColumn
         * @returns {Promise<Trie?>}
         */
        async search(name, dataColumn) {
            if (typeof name === "string") {
                const utf8encoder = new TextEncoder();
                name = utf8encoder.encode(name);
            }
            const searchPattern = new Uint8ArraySearchPattern(name);
            let trie = this.trie(dataColumn, searchPattern);
            for (const datum of name) {
                // code point definitely exists
                const newTrie = trie.child(datum);
                if (newTrie) {
                    trie = await newTrie;
                } else {
                    return null;
                }
            }
            return trie;
        }

        /**
         * @param {Uint8Array|string} _name
         * @param {DataColumn} _dataColumn
         * @returns {AsyncGenerator<Trie>}
         */
        async* searchLev(_name, _dataColumn) {
            // this function only returns whole-string matches,
            // which pure-suffix nodes don't have, so is
            // intentionally blank
        }

        /** @returns {RoaringBitmap} */
        getCurrentLeaves() {
            return this.leaves_suffix;
        }
    }

    /**
     * A representation of a set of strings in the search index,
     * as a subset of the entire tree (suffix-only).
     */
    class SuffixTrie {
        /**
         * @param {SuffixSearchTree} tree
         * @param {number} offset
         * @param {DataColumn} dataColumn
         * @param {Uint8ArraySearchPattern} searchPattern
         */
        constructor(tree, offset, dataColumn, searchPattern) {
            this.tree = tree;
            this.offset = offset;
            this.dataColumn = dataColumn;
            this.searchPattern = searchPattern;
        }

        /**
         * All exact matches for the string represented by this node.
         * Since pure-suffix nodes have no exactly-matching children,
         * this function returns the empty bitmap.
         * @returns {RoaringBitmap}
         */
        matches() {
            return EMPTY_BITMAP;
        }

        /**
         * All matches for strings that contain the string represented by this node.
         * @returns {AsyncGenerator<RoaringBitmap>}
         */
        async* substringMatches() {
            /** @type {Promise<SearchTree>[]} */
            let layer = [Promise.resolve(this.tree)];
            while (layer.length) {
                const current_layer = layer;
                layer = [];
                for await (const tree of current_layer) {
                    /** @type {number[]?} */
                    let rejected = null;
                    let leaves = tree.getCurrentLeaves();
                    for (const leaf of leaves.entries()) {
                        const haystack = await this.dataColumn.at(leaf);
                        if (haystack === undefined || !this.searchPattern.matches(haystack)) {
                            if (!rejected) {
                                rejected = [];
                            }
                            rejected.push(leaf);
                        }
                    }
                    if (rejected) {
                        if (leaves.cardinality() !== rejected.length) {
                            for (const rej of rejected) {
                                leaves = leaves.remove(rej);
                            }
                            yield leaves;
                        }
                    } else {
                        yield leaves;
                    }
                }
                /** @type {HashTable<[number, SearchTree][]>} */
                const subnodes = new HashTable();
                for await (const node of current_layer) {
                    const branches = node.branches;
                    const l = branches.subtrees.length;
                    for (let i = 0; i < l; ++i) {
                        const subtree = branches.subtrees[i];
                        if (subtree) {
                            layer.push(subtree);
                        } else if (subtree === null) {
                            const newnode = branches.getNodeID(i);
                            if (!newnode) {
                                throw new Error(`malformed tree; no node for index ${i}`);
                            } else {
                                let subnode_list = subnodes.get(newnode);
                                if (!subnode_list) {
                                    subnode_list = [[i, node]];
                                    subnodes.set(newnode, subnode_list);
                                } else {
                                    subnode_list.push([i, node]);
                                }
                            }
                        } else {
                            throw new Error(`malformed tree; index ${i} does not exist`);
                        }
                    }
                }
                for (const [newnode, subnode_list] of subnodes.entries()) {
                    const res = registry.searchTreeLoadByNodeID(newnode);
                    for (const [i, node] of subnode_list) {
                        const branches = node.branches;
                        branches.subtrees[i] = res;
                    }
                    layer.push(res);
                }
            }
        }

        /**
         * All matches for strings that start with the string represented by this node.
         * Since this is a pure-suffix node, there aren't any.
         * @returns {AsyncGenerator<RoaringBitmap>}
         */
        async* prefixMatches() {
            // this function only returns prefix matches,
            // which pure-suffix nodes don't have, so is
            // intentionally blank
        }

        /**
         * Returns all keys that are children of this node.
         * @returns {Uint8Array}
         */
        keysExcludeSuffixOnly() {
            return EMPTY_UINT8;
        }

        /**
         * Returns all nodes that are direct children of this node.
         * @returns {[number, Promise<Trie>][]}
         */
        childrenExcludeSuffixOnly() {
            return [];
        }

        /**
         * Returns a single node that is a direct child of this node.
         * @param {number} byte
         * @returns {Promise<Trie>?}
         */
        child(byte) {
            if (this.offset === this.tree.dataLen) {
                const i = this.tree.branches.getIndex(byte);
                if (i !== -1) {
                    /** @type {Promise<SearchTree>?} */
                    let branch = this.tree.branches.subtrees[i];
                    if (branch === null) {
                        const newnode = this.tree.branches.getNodeID(i);
                        if (!newnode) {
                            throw new Error(`malformed tree; no node for key ${byte}`);
                        }
                        branch = registry.searchTreeLoadByNodeID(newnode);
                        this.tree.branches.subtrees[i] = branch;
                    }
                    return branch.then(branch => branch.trie(this.dataColumn, this.searchPattern));
                }
            } else {
                return Promise.resolve(new SuffixTrie(
                    this.tree,
                    this.offset + 1,
                    this.dataColumn,
                    this.searchPattern,
                ));
            }
            return null;
        }
    }

    class DataColumn {
        /**
         * Construct the wrapper object for a data column.
         * @param {number[]} counts
         * @param {Uint8Array} hashes
         * @param {RoaringBitmap} emptyset
         * @param {string} name
         * @param {SearchTree?} searchTree
         */
        constructor(counts, hashes, emptyset, name, searchTree) {
            this.searchTree = searchTree;
            this.hashes = hashes;
            this.emptyset = emptyset;
            this.name = name;
            /** @type {{"hash": Uint8Array, "data": Promise<Uint8Array[]>?, "end": number}[]} */
            this.buckets = [];
            this.bucket_keys = [];
            const l = counts.length;
            let k = 0;
            let totalLength = 0;
            for (let i = 0; i < l; ++i) {
                const count = counts[i];
                totalLength += count;
                const start = k;
                for (let j = 0; j < count; ++j) {
                    if (emptyset.contains(k)) {
                        j -= 1;
                    }
                    k += 1;
                }
                const end = k;
                const bucket = {hash: hashes.subarray(i * 6, (i + 1) * 6), data: null, end, count};
                this.buckets.push(bucket);
                this.bucket_keys.push(start);
            }
            this.length = totalLength;
        }
        /**
         * Check if a cell contains the empty string.
         * @param {number} id
         * @returns {boolean}
         */
        isEmpty(id) {
            return this.emptyset.contains(id);
        }
        /**
         * Look up a cell by row ID.
         * @param {number} id
         * @returns {Promise<Uint8Array|undefined>}
         */
        async at(id) {
            if (this.emptyset.contains(id)) {
                return Promise.resolve(EMPTY_UINT8);
            } else {
                let idx = -1;
                while (this.bucket_keys[idx + 1] <= id) {
                    idx += 1;
                }
                if (idx === -1 || idx >= this.bucket_keys.length) {
                    return Promise.resolve(undefined);
                } else {
                    const start = this.bucket_keys[idx];
                    const {hash, end} = this.buckets[idx];
                    let data = this.buckets[idx].data;
                    if (data === null) {
                        const dataSansEmptysetOrig = await registry.dataLoadByNameAndHash(
                            this.name,
                            hash,
                        );
                        // After the `await` resolves, another task might fill
                        // in the data. If so, we should use that.
                        data = this.buckets[idx].data;
                        if (data !== null) {
                            return (await data)[id - start];
                        }
                        const dataSansEmptyset = [...dataSansEmptysetOrig];
                        /** @type {(Uint8Array[])|null} */
                        let dataWithEmptyset = null;
                        let pos = start;
                        let insertCount = 0;
                        while (pos < end) {
                            if (this.emptyset.contains(pos)) {
                                if (dataWithEmptyset === null) {
                                    dataWithEmptyset = dataSansEmptyset.splice(0, insertCount);
                                } else if (insertCount !== 0) {
                                    dataWithEmptyset.push(
                                        ...dataSansEmptyset.splice(0, insertCount),
                                    );
                                }
                                insertCount = 0;
                                dataWithEmptyset.push(EMPTY_UINT8);
                            } else {
                                insertCount += 1;
                            }
                            pos += 1;
                        }
                        data = Promise.resolve(
                            dataWithEmptyset === null ?
                                dataSansEmptyset :
                                dataWithEmptyset.concat(dataSansEmptyset),
                        );
                        this.buckets[idx].data = data;
                    }
                    return (await data)[id - start];
                }
            }
        }
        /**
         * Search by exact substring
         * @param {Uint8Array|string} name
         * @returns {Promise<Trie?>}
         */
        async search(name) {
            return await (this.searchTree ? this.searchTree.search(name, this) : null);
        }
        /**
         * Search by whole, inexact match
         * @param {Uint8Array|string} name
         * @returns {AsyncGenerator<Trie>}
         */
        async *searchLev(name) {
            if (this.searchTree) {
                yield* this.searchTree.searchLev(name, this);
            }
        }
    }

    class Database {
        /**
         * The primary frontend for accessing data in this index.
         *
         * @param {Map<string, SearchTree>} searchTreeRoots
         * @param {Map<string, DataColumn>} dataColumns
         */
        constructor(searchTreeRoots, dataColumns) {
            this.searchTreeRoots = searchTreeRoots;
            this.dataColumns = dataColumns;
        }
        /**
         * Search a column by name, returning verbatim matched IDs.
         * @param {string} colname
         * @returns {SearchTree|undefined}
         */
        getIndex(colname) {
            return this.searchTreeRoots.get(colname);
        }
        /**
         * Look up a cell by column ID and row ID.
         * @param {string} colname
         * @returns {DataColumn|undefined}
         */
        getData(colname) {
            return this.dataColumns.get(colname);
        }
    }

    /**
     * Load a data column.
     * @param {Uint8Array} data
     */
    function loadColumnFromBytes(data) {
        const hashBuf = Uint8Array.of(0, 0, 0, 0, 0, 0, 0, 0);
        const truncatedHash = hashBuf.subarray(2, 8);
        siphashOfBytes(data, 0, 0, 0, 0, hashBuf);
        const cb = registry.dataColumnLoadPromiseCallbacks.get(truncatedHash);
        if (cb) {
            const backrefs = [];
            const dataSansEmptyset = [];
            let i = 0;
            const l = data.length;
            while (i < l) {
                let c = data[i];
                if (c >= 48 && c <= 63) { // 48 = "0", 63 = "?"
                    dataSansEmptyset.push(backrefs[c - 48]);
                    i += 1;
                } else {
                    let n = 0;
                    while (c < 96) { // 96 = "`"
                        n = (n << 4) | (c & 0xF);
                        i += 1;
                        c = data[i];
                    }
                    n = (n << 4) | (c & 0xF);
                    i += 1;
                    const item = data.subarray(i, i + n);
                    dataSansEmptyset.push(item);
                    i += n;
                    backrefs.unshift(item);
                    if (backrefs.length > 16) {
                        backrefs.pop();
                    }
                }
            }
            cb(null, dataSansEmptyset);
        }
    }

    /**
     * @param {string} inputBase64
     * @returns {[Uint8Array, SearchTree]}
     */
    function makeSearchTreeFromBase64(inputBase64) {
        const input = makeUint8ArrayFromBase64(inputBase64);
        let i = 0;
        const l = input.length;
        /** @type {HashTable<SearchTree>} */
        const stash = new HashTable();
        const hash = Uint8Array.of(0, 0, 0, 0, 0, 0, 0, 0);
        const truncatedHash = new Uint8Array(hash.buffer, 2, 6);
        // used for handling compressed (that is, relative-offset) nodes
        /** @type {{hash: Uint8Array, used: boolean}[]} */
        const hash_history = [];
        /** @type {Uint8Array[]} */
        const data_history = [];
        let canonical = EMPTY_UINT8;
        /** @type {SearchTree} */
        let tree = new PrefixSearchTree(
            EMPTY_SEARCH_TREE_BRANCHES,
            EMPTY_SEARCH_TREE_BRANCHES,
            EMPTY_UINT8,
            EMPTY_BITMAP,
            EMPTY_BITMAP,
        );
        /**
         * @param {Uint8Array} input
         * @param {number} i
         * @param {number} compression_tag
         * @returns {{
         *     "cpbranches": Uint8Array,
         *     "csbranches": Uint8Array,
         *     "might_have_prefix_branches": SearchTreeBranches<SearchTree>,
         *     "branches": SearchTreeBranches<SearchTree>,
         *     "cpnodes": Uint8Array,
         *     "csnodes": Uint8Array,
         *     "consumed_len_bytes": number,
         * }}
         */
        function makeBranchesFromBinaryData(
            input,
            i,
            compression_tag,
        ) {
            const is_pure_suffixes_only_node = (compression_tag & 0x01) !== 0x00;
            const is_stack_compressed = (compression_tag & 0x02) !== 0;
            const is_long_compressed = (compression_tag & 0x04) !== 0;
            const all_children_are_compressed =
                (compression_tag & 0xF0) === 0xF0 && !is_long_compressed;
            const any_children_are_compressed =
                (compression_tag & 0xF0) !== 0x00 || is_long_compressed;
            const start_point = i;
            let cplen;
            let cslen;
            /**
             * @type {(
             *   typeof SearchTreeBranches<SearchTree> &
             *   {"ALPHABITMAP_CHARS": number[], "width": number}
             * )?}
             */
            let alphabitmap = null;
            if (is_pure_suffixes_only_node) {
                cplen = 0;
                cslen = input[i];
                i += 1;
                if (cslen >= 0xc0) {
                    alphabitmap = SearchTreeBranchesLongAlphaBitmap;
                    cslen = cslen & 0x3F;
                } else if (cslen >= 0x80) {
                    alphabitmap = SearchTreeBranchesShortAlphaBitmap;
                    cslen = cslen & 0x7F;
                }
            } else {
                cplen = input[i];
                i += 1;
                cslen = input[i];
                i += 1;
                if (cplen === 0xff && cslen === 0xff) {
                    cplen = 0x100;
                    cslen = 0;
                } else if (cplen >= 0xc0 && cslen >= 0xc0) {
                    alphabitmap = SearchTreeBranchesLongAlphaBitmap;
                    cplen = cplen & 0x3F;
                    cslen = cslen & 0x3F;
                } else if (cplen >= 0x80 && cslen >= 0x80) {
                    alphabitmap = SearchTreeBranchesShortAlphaBitmap;
                    cplen = cplen & 0x7F;
                    cslen = cslen & 0x7F;
                }
            }
            let j = 0;
            /** @type {Uint8Array} */
            let cpnodes;
            if (any_children_are_compressed) {
                cpnodes = cplen === 0 ? EMPTY_UINT8 : new Uint8Array(cplen * 6);
                while (j < cplen) {
                    const is_compressed = all_children_are_compressed ||
                        ((0x10 << j) & compression_tag) !== 0;
                    if (is_compressed) {
                        let slot = hash_history.length - 1;
                        if (is_stack_compressed) {
                            while (hash_history[slot].used) {
                                slot -= 1;
                            }
                        } else {
                            slot -= input[i];
                            i += 1;
                        }
                        hash_history[slot].used = true;
                        cpnodes.set(
                            hash_history[slot].hash,
                            j * 6,
                        );
                    } else {
                        const joff = j * 6;
                        cpnodes[joff + 0] = input[i + 0];
                        cpnodes[joff + 1] = input[i + 1];
                        cpnodes[joff + 2] = input[i + 2];
                        cpnodes[joff + 3] = input[i + 3];
                        cpnodes[joff + 4] = input[i + 4];
                        cpnodes[joff + 5] = input[i + 5];
                        i += 6;
                    }
                    j += 1;
                }
            } else {
                cpnodes = cplen === 0 ? EMPTY_UINT8 : input.subarray(i, i + (cplen * 6));
                i += cplen * 6;
            }
            j = 0;
            /** @type {Uint8Array} */
            let csnodes;
            if (any_children_are_compressed) {
                csnodes = cslen === 0 ? EMPTY_UINT8 : new Uint8Array(cslen * 6);
                while (j < cslen) {
                    const is_compressed = all_children_are_compressed ||
                        ((0x10 << (cplen + j)) & compression_tag) !== 0;
                    if (is_compressed) {
                        let slot = hash_history.length - 1;
                        if (is_stack_compressed) {
                            while (hash_history[slot].used) {
                                slot -= 1;
                            }
                        } else {
                            slot -= input[i];
                            i += 1;
                        }
                        hash_history[slot].used = true;
                        csnodes.set(
                            hash_history[slot].hash,
                            j * 6,
                        );
                    } else {
                        const joff = j * 6;
                        csnodes[joff + 0] = input[i + 0];
                        csnodes[joff + 1] = input[i + 1];
                        csnodes[joff + 2] = input[i + 2];
                        csnodes[joff + 3] = input[i + 3];
                        csnodes[joff + 4] = input[i + 4];
                        csnodes[joff + 5] = input[i + 5];
                        i += 6;
                    }
                    j += 1;
                }
            } else {
                csnodes = cslen === 0 ? EMPTY_UINT8 : input.subarray(i, i + (cslen * 6));
                i += cslen * 6;
            }
            let cpbranches;
            let might_have_prefix_branches;
            if (cplen === 0) {
                cpbranches = EMPTY_UINT8;
                might_have_prefix_branches = EMPTY_SEARCH_TREE_BRANCHES;
            } else if (alphabitmap) {
                cpbranches = new Uint8Array(input.buffer, i + input.byteOffset, alphabitmap.width);
                const branchset = (alphabitmap.width === 4 ? (input[i + 3] << 24) : 0) |
                    (input[i + 2] << 16) |
                    (input[i + 1] << 8) |
                    input[i];
                might_have_prefix_branches = new alphabitmap(branchset, cpnodes);
                i += alphabitmap.width;
            } else {
                cpbranches = new Uint8Array(input.buffer, i + input.byteOffset, cplen);
                might_have_prefix_branches = new SearchTreeBranchesArray(cpbranches, cpnodes);
                i += cplen;
            }
            let csbranches;
            let branches;
            if (cslen === 0) {
                csbranches = EMPTY_UINT8;
                branches = might_have_prefix_branches;
            } else if (alphabitmap) {
                csbranches = new Uint8Array(input.buffer, i + input.byteOffset, alphabitmap.width);
                const branchset = (alphabitmap.width === 4 ? (input[i + 3] << 24) : 0) |
                    (input[i + 2] << 16) |
                    (input[i + 1] << 8) |
                    input[i];
                if (cplen === 0) {
                    branches = new alphabitmap(branchset, csnodes);
                } else {
                    const cpoffset = i - alphabitmap.width;
                    const cpbranchset =
                        (alphabitmap.width === 4 ? (input[cpoffset + 3] << 24) : 0) |
                        (input[cpoffset + 2] << 16) |
                        (input[cpoffset + 1] << 8) |
                        input[cpoffset];
                    const hashes = new Uint8Array((cplen + cslen) * 6);
                    let cpi = 0;
                    let csi = 0;
                    let j = 0;
                    for (let k = 0; k < alphabitmap.ALPHABITMAP_CHARS.length; k += 1) {
                        if (branchset & (1 << k)) {
                            hashes[j + 0] = csnodes[csi + 0];
                            hashes[j + 1] = csnodes[csi + 1];
                            hashes[j + 2] = csnodes[csi + 2];
                            hashes[j + 3] = csnodes[csi + 3];
                            hashes[j + 4] = csnodes[csi + 4];
                            hashes[j + 5] = csnodes[csi + 5];
                            j += 6;
                            csi += 6;
                        } else if (cpbranchset & (1 << k)) {
                            hashes[j + 0] = cpnodes[cpi + 0];
                            hashes[j + 1] = cpnodes[cpi + 1];
                            hashes[j + 2] = cpnodes[cpi + 2];
                            hashes[j + 3] = cpnodes[cpi + 3];
                            hashes[j + 4] = cpnodes[cpi + 4];
                            hashes[j + 5] = cpnodes[cpi + 5];
                            j += 6;
                            cpi += 6;
                        }
                    }
                    branches = new alphabitmap(branchset | cpbranchset, hashes);
                }
                i += alphabitmap.width;
            } else {
                csbranches = new Uint8Array(input.buffer, i + input.byteOffset, cslen);
                if (cplen === 0) {
                    branches = new SearchTreeBranchesArray(csbranches, csnodes);
                } else {
                    const branchset = new Uint8Array(cplen + cslen);
                    const hashes = new Uint8Array((cplen + cslen) * 6);
                    let cpi = 0;
                    let csi = 0;
                    let j = 0;
                    while (cpi < cplen || csi < cslen) {
                        if (cpi >= cplen || (csi < cslen && cpbranches[cpi] > csbranches[csi])) {
                            branchset[j] = csbranches[csi];
                            const joff = j * 6;
                            const csioff = csi * 6;
                            hashes[joff + 0] = csnodes[csioff + 0];
                            hashes[joff + 1] = csnodes[csioff + 1];
                            hashes[joff + 2] = csnodes[csioff + 2];
                            hashes[joff + 3] = csnodes[csioff + 3];
                            hashes[joff + 4] = csnodes[csioff + 4];
                            hashes[joff + 5] = csnodes[csioff + 5];
                            csi += 1;
                        } else {
                            branchset[j] = cpbranches[cpi];
                            const joff = j * 6;
                            const cpioff = cpi * 6;
                            hashes[joff + 0] = cpnodes[cpioff + 0];
                            hashes[joff + 1] = cpnodes[cpioff + 1];
                            hashes[joff + 2] = cpnodes[cpioff + 2];
                            hashes[joff + 3] = cpnodes[cpioff + 3];
                            hashes[joff + 4] = cpnodes[cpioff + 4];
                            hashes[joff + 5] = cpnodes[cpioff + 5];
                            cpi += 1;
                        }
                        j += 1;
                    }
                    branches = new SearchTreeBranchesArray(branchset, hashes);
                }
                i += cslen;
            }
            return {
                consumed_len_bytes: i - start_point,
                cpbranches,
                csbranches,
                cpnodes,
                csnodes,
                branches,
                might_have_prefix_branches,
            };
        }
        while (i < l) {
            const start = i;
            let data;
            // compression_tag = 1 means pure-suffixes-only,
            // which is not considered "compressed" for the purposes of this loop
            // because that's the canonical, hashed version of the data
            let compression_tag = input[i];
            const is_pure_suffixes_only_node = (compression_tag & 0x01) !== 0;
            if (compression_tag > 1) {
                // compressed node
                const is_long_compressed = (compression_tag & 0x04) !== 0;
                const is_data_compressed = (compression_tag & 0x08) !== 0;
                i += 1;
                if (is_long_compressed) {
                    compression_tag |= input[i] << 8;
                    i += 1;
                    compression_tag |= input[i] << 16;
                    i += 1;
                }
                let dlen = input[i];
                i += 1;
                if (is_data_compressed) {
                    data = data_history[data_history.length - dlen - 1];
                    dlen = data.length;
                } else if (is_pure_suffixes_only_node) {
                    data = EMPTY_UINT8;
                } else {
                    data = dlen === 0 ?
                        EMPTY_UINT8 :
                        new Uint8Array(input.buffer, i + input.byteOffset, dlen);
                    i += dlen;
                }
                const coffset = i;
                const {
                    cpbranches,
                    csbranches,
                    cpnodes,
                    csnodes,
                    consumed_len_bytes: branches_consumed_len_bytes,
                    branches,
                    might_have_prefix_branches,
                } = makeBranchesFromBinaryData(input, i, compression_tag);
                i += branches_consumed_len_bytes;
                let whole;
                let suffix;
                if (is_pure_suffixes_only_node) {
                    suffix = input[i] === 0 ?
                        EMPTY_BITMAP1 :
                        new RoaringBitmap(input, i);
                    i += suffix.consumed_len_bytes;
                    tree = new SuffixSearchTree(
                        branches,
                        dlen,
                        suffix,
                    );
                    const clen = (
                        3 + // lengths of children and data
                        csnodes.length +
                        csbranches.length +
                        suffix.consumed_len_bytes
                    );
                    if (canonical.length < clen) {
                        canonical = new Uint8Array(clen);
                    }
                    let ci = 0;
                    canonical[ci] = 1;
                    ci += 1;
                    canonical[ci] = dlen;
                    ci += 1;
                    canonical[ci] = input[coffset]; // suffix child count
                    ci += 1;
                    canonical.set(csnodes, ci);
                    ci += csnodes.length;
                    canonical.set(csbranches, ci);
                    ci += csbranches.length;
                    const leavesOffset = i - suffix.consumed_len_bytes;
                    for (let j = leavesOffset; j < i; j += 1) {
                        canonical[ci + j - leavesOffset] = input[j];
                    }
                    siphashOfBytes(canonical.subarray(0, clen), 0, 0, 0, 0, hash);
                } else {
                    if (input[i] === 0xff) {
                        whole = EMPTY_BITMAP;
                        suffix = EMPTY_BITMAP1;
                        i += 1;
                    } else {
                        whole = input[i] === 0 ?
                            EMPTY_BITMAP1 :
                            new RoaringBitmap(input, i);
                        i += whole.consumed_len_bytes;
                        suffix = input[i] === 0 ?
                            EMPTY_BITMAP1 :
                            new RoaringBitmap(input, i);
                        i += suffix.consumed_len_bytes;
                    }
                    tree = new PrefixSearchTree(
                        branches,
                        might_have_prefix_branches,
                        data,
                        whole,
                        suffix,
                    );
                    const clen = (
                        4 + // lengths of children and data
                        dlen +
                        cpnodes.length + csnodes.length +
                        cpbranches.length + csbranches.length +
                        whole.consumed_len_bytes +
                        suffix.consumed_len_bytes
                    );
                    if (canonical.length < clen) {
                        canonical = new Uint8Array(clen);
                    }
                    let ci = 0;
                    canonical[ci] = 0;
                    ci += 1;
                    canonical[ci] = dlen;
                    ci += 1;
                    canonical.set(data, ci);
                    ci += data.length;
                    canonical[ci] = input[coffset]; // prefix child count
                    ci += 1;
                    canonical[ci] = input[coffset + 1]; // suffix child count
                    ci += 1;
                    canonical.set(cpnodes, ci);
                    ci += cpnodes.length;
                    canonical.set(csnodes, ci);
                    ci += csnodes.length;
                    canonical.set(cpbranches, ci);
                    ci += cpbranches.length;
                    canonical.set(csbranches, ci);
                    ci += csbranches.length;
                    const leavesOffset = i - whole.consumed_len_bytes - suffix.consumed_len_bytes;
                    for (let j = leavesOffset; j < i; j += 1) {
                        canonical[ci + j - leavesOffset] = input[j];
                    }
                    siphashOfBytes(canonical.subarray(0, clen), 0, 0, 0, 0, hash);
                }
                hash[2] &= 0x7f;
            } else {
                // uncompressed node
                const dlen = input [i + 1];
                i += 2;
                if (dlen === 0 || is_pure_suffixes_only_node) {
                    data = EMPTY_UINT8;
                } else {
                    data = new Uint8Array(input.buffer, i + input.byteOffset, dlen);
                    i += dlen;
                }
                const {
                    consumed_len_bytes: branches_consumed_len_bytes,
                    branches,
                    might_have_prefix_branches,
                } = makeBranchesFromBinaryData(input, i, compression_tag);
                i += branches_consumed_len_bytes;
                let whole;
                let suffix;
                if (is_pure_suffixes_only_node) {
                    whole = EMPTY_BITMAP;
                    suffix = input[i] === 0 ?
                        EMPTY_BITMAP1 :
                        new RoaringBitmap(input, i);
                    i += suffix.consumed_len_bytes;
                } else if (input[i] === 0xff) {
                    whole = EMPTY_BITMAP;
                    suffix = EMPTY_BITMAP;
                    i += 1;
                } else {
                    whole = input[i] === 0 ?
                        EMPTY_BITMAP1 :
                        new RoaringBitmap(input, i);
                    i += whole.consumed_len_bytes;
                    suffix = input[i] === 0 ?
                        EMPTY_BITMAP1 :
                        new RoaringBitmap(input, i);
                    i += suffix.consumed_len_bytes;
                }
                siphashOfBytes(new Uint8Array(
                    input.buffer,
                    start + input.byteOffset,
                    i - start,
                ), 0, 0, 0, 0, hash);
                hash[2] &= 0x7f;
                tree = is_pure_suffixes_only_node ?
                    new SuffixSearchTree(
                        branches,
                        dlen,
                        suffix,
                    ) :
                    new PrefixSearchTree(
                        branches,
                        might_have_prefix_branches,
                        data,
                        whole,
                        suffix,
                    );
            }
            hash_history.push({hash: truncatedHash.slice(), used: false});
            if (data.length !== 0) {
                data_history.push(data);
            }
            const tree_branch_nodeids = tree.branches.nodeids;
            const tree_branch_subtrees = tree.branches.subtrees;
            let j = 0;
            let lb = tree.branches.subtrees.length;
            while (j < lb) {
                // node id with a 1 in its most significant bit is inlined, and, so
                // it won't be in the stash
                if ((tree_branch_nodeids[j * 6] & 0x80) === 0) {
                    const subtree = stash.getWithOffsetKey(tree_branch_nodeids, j * 6);
                    if (subtree !== undefined) {
                        tree_branch_subtrees[j] = Promise.resolve(subtree);
                    }
                }
                j += 1;
            }
            if (tree instanceof PrefixSearchTree) {
                const tree_mhp_branch_nodeids = tree.might_have_prefix_branches.nodeids;
                const tree_mhp_branch_subtrees = tree.might_have_prefix_branches.subtrees;
                j = 0;
                lb = tree.might_have_prefix_branches.subtrees.length;
                while (j < lb) {
                    // node id with a 1 in its most significant bit is inlined, and, so
                    // it won't be in the stash
                    if ((tree_mhp_branch_nodeids[j * 6] & 0x80) === 0) {
                        const subtree = stash.getWithOffsetKey(tree_mhp_branch_nodeids, j * 6);
                        if (subtree !== undefined) {
                            tree_mhp_branch_subtrees[j] = Promise.resolve(subtree);
                        }
                    }
                    j += 1;
                }
            }
            if (i !== l) {
                stash.set(truncatedHash, tree);
            }
        }
        return [truncatedHash, tree];
    }

    return new Promise((resolve, reject) => {
        registry.searchTreeRootCallback = (error, data) => {
            if (data) {
                resolve(data);
            } else {
                reject(error);
            }
        };
        hooks.loadRoot(callbacks);
    });
}

if (typeof window !== "undefined") {
    window.Stringdex = {
        loadDatabase,
    };
    window.RoaringBitmap = RoaringBitmap;
    if (window.StringdexOnload) {
        window.StringdexOnload.forEach(cb => cb(window.Stringdex));
    }
} else {
    /** @type {stringdex.Stringdex} */
    // eslint-disable-next-line no-undef
    module.exports.Stringdex = {
        loadDatabase,
    };
    /** @type {stringdex.RoaringBitmap} */
    // eslint-disable-next-line no-undef
    module.exports.RoaringBitmap = RoaringBitmap;
}

// eslint-disable-next-line max-len
// polyfill https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Uint8Array/fromBase64
/**
 * @type {function(string): Uint8Array} base64
 */
//@ts-expect-error
const makeUint8ArrayFromBase64 = Uint8Array.fromBase64 ? Uint8Array.fromBase64 : (string => {
    const bytes_as_string = atob(string);
    const l = bytes_as_string.length;
    const bytes = new Uint8Array(l);
    for (let i = 0; i < l; ++i) {
        bytes[i] = bytes_as_string.charCodeAt(i);
    }
    return bytes;
});
/**
 * @type {function(string): Uint8Array} base64
 */
//@ts-expect-error
const makeUint8ArrayFromHex = Uint8Array.fromHex ? Uint8Array.fromHex : (string => {
    /** @type {Object<string, number>} */
    const alpha = {
        "0": 0, "1": 1,
        "2": 2, "3": 3,
        "4": 4, "5": 5,
        "6": 6, "7": 7,
        "8": 8, "9": 9,
        "a": 10, "b": 11,
        "A": 10, "B": 11,
        "c": 12, "d": 13,
        "C": 12, "D": 13,
        "e": 14, "f": 15,
        "E": 14, "F": 15,
    };
    const l = string.length >> 1;
    const bytes = new Uint8Array(l);
    for (let i = 0; i < l; i += 1) {
        const top = string[i << 1];
        const bottom = string[(i << 1) + 1];
        bytes[i] = (alpha[top] << 4) | alpha[bottom];
    }
    return bytes;
});

/**
 * @type {function(Uint8Array): string} base64
 */
//@ts-expect-error
const makeHexFromUint8Array = Uint8Array.prototype.toHex ? (array => array.toHex()) : (array => {
    /** @type {string[]} */
    const alpha = [
        "0", "1",
        "2", "3",
        "4", "5",
        "6", "7",
        "8", "9",
        "a", "b",
        "c", "d",
        "e", "f",
    ];
    const l = array.length;
    const v = [];
    for (let i = 0; i < l; ++i) {
        v.push(alpha[array[i] >> 4]);
        v.push(alpha[array[i] & 0xf]);
    }
    return v.join("");
});

//////////////

/**
 * SipHash 1-3
 * @param {Uint8Array} input data to be hashed; all codepoints in string should be less than 256
 * @param {number} k0lo first word of key
 * @param {number} k0hi second word of key
 * @param {number} k1lo third word of key
 * @param {number} k1hi fourth word of key
 * @param {Uint8Array} output the data to write (clobber the first eight bytes)
 */
function siphashOfBytes(input, k0lo, k0hi, k1lo, k1hi, output) {
    // hash state
    // While siphash uses 64 bit state, js only has native support
    // for 32 bit numbers. BigInt, unfortunately, doesn't count.
    // It's too slow.
    let v0lo = k0lo ^ 0x70736575;
    let v0hi = k0hi ^ 0x736f6d65;
    let v1lo = k1lo ^ 0x6e646f6d;
    let v1hi = k1hi ^ 0x646f7261;
    let v2lo = k0lo ^ 0x6e657261;
    let v2hi = k0hi ^ 0x6c796765;
    let v3lo = k1lo ^ 0x79746573;
    let v3hi = k1hi ^ 0x74656462;
    const inputLength = input.length;
    let inputI = 0;
    // main hash loop
    const left = inputLength & 0x7;
    let milo = 0;
    let mihi = 0;
    while (inputI < inputLength - left) {
        u8ToU64le(inputI, inputI + 8);
        v3lo ^= milo;
        v3hi ^= mihi;
        siphashCompress();
        v0lo ^= milo;
        v0hi ^= mihi;
        inputI += 8;
    }
    u8ToU64le(inputI, inputI + left);
    // finish
    const blo = milo;
    const bhi = ((inputLength & 0xff) << 24) | mihi;
    v3lo ^= blo;
    v3hi ^= bhi;
    siphashCompress();
    v0lo ^= blo;
    v0hi ^= bhi;
    v2lo ^= 0xff;
    siphashCompress();
    siphashCompress();
    siphashCompress();
    output[7] = (v0lo ^ v1lo ^ v2lo ^ v3lo) & 0xff;
    output[6] = (v0lo ^ v1lo ^ v2lo ^ v3lo) >>> 8;
    output[5] = (v0lo ^ v1lo ^ v2lo ^ v3lo) >>> 16;
    output[4] = (v0lo ^ v1lo ^ v2lo ^ v3lo) >>> 24;
    output[3] = (v0hi ^ v1hi ^ v2hi ^ v3hi) & 0xff;
    output[2] = (v0hi ^ v1hi ^ v2hi ^ v3hi) >>> 8;
    output[1] = (v0hi ^ v1hi ^ v2hi ^ v3hi) >>> 16;
    output[0] = (v0hi ^ v1hi ^ v2hi ^ v3hi) >>> 24;
    /**
     * Convert eight bytes to a single 64-bit number
     * @param {number} offset
     * @param {number} length
     */
    function u8ToU64le(offset, length) {
        const n0 = offset < length ? input[offset] & 0xff : 0;
        const n1 = offset + 1 < length ? input[offset + 1] & 0xff : 0;
        const n2 = offset + 2 < length ? input[offset + 2] & 0xff : 0;
        const n3 = offset + 3 < length ? input[offset + 3] & 0xff : 0;
        const n4 = offset + 4 < length ? input[offset + 4] & 0xff : 0;
        const n5 = offset + 5 < length ? input[offset + 5] & 0xff : 0;
        const n6 = offset + 6 < length ? input[offset + 6] & 0xff : 0;
        const n7 = offset + 7 < length ? input[offset + 7] & 0xff : 0;
        milo = n0 | (n1 << 8) | (n2 << 16) | (n3 << 24);
        mihi = n4 | (n5 << 8) | (n6 << 16) | (n7 << 24);
    }
    function siphashCompress() {
        // v0 += v1;
        v0hi = (v0hi + v1hi + (((v0lo >>> 0) + (v1lo >>> 0) > 0xffffffff) ? 1 : 0)) | 0;
        v0lo = (v0lo + v1lo) | 0;
        // rotl(v1, 13)
        let v1lo_ = v1lo;
        let v1hi_ = v1hi;
        v1lo = (v1lo_ << 13) | (v1hi_ >>> 19);
        v1hi = (v1hi_ << 13) | (v1lo_ >>> 19);
        // v1 ^= v0
        v1lo ^= v0lo;
        v1hi ^= v0hi;
        // rotl(v0, 32)
        const v0lo_ = v0lo;
        const v0hi_ = v0hi;
        v0lo = v0hi_;
        v0hi = v0lo_;
        // v2 += v3
        v2hi = (v2hi + v3hi + (((v2lo >>> 0) + (v3lo >>> 0) > 0xffffffff) ? 1 : 0)) | 0;
        v2lo = (v2lo + v3lo) | 0;
        // rotl(v3, 16)
        let v3lo_ = v3lo;
        let v3hi_ = v3hi;
        v3lo = (v3lo_ << 16) | (v3hi_ >>> 16);
        v3hi = (v3hi_ << 16) | (v3lo_ >>> 16);
        // v3 ^= v2
        v3lo ^= v2lo;
        v3hi ^= v2hi;
        // v0 += v3
        v0hi = (v0hi + v3hi + (((v0lo >>> 0) + (v3lo >>> 0) > 0xffffffff) ? 1 : 0)) | 0;
        v0lo = (v0lo + v3lo) | 0;
        // rotl(v3, 21)
        v3lo_ = v3lo;
        v3hi_ = v3hi;
        v3lo = (v3lo_ << 21) | (v3hi_ >>> 11);
        v3hi = (v3hi_ << 21) | (v3lo_ >>> 11);
        // v3 ^= v0
        v3lo ^= v0lo;
        v3hi ^= v0hi;
        // v2 += v1
        v2hi = (v2hi + v1hi + (((v2lo >>> 0) + (v1lo >>> 0) > 0xffffffff) ? 1 : 0)) | 0;
        v2lo = (v2lo + v1lo) | 0;
        // rotl(v1, 17)
        v1lo_ = v1lo;
        v1hi_ = v1hi;
        v1lo = (v1lo_ << 17) | (v1hi_ >>> 15);
        v1hi = (v1hi_ << 17) | (v1lo_ >>> 15);
        // v1 ^= v2
        v1lo ^= v2lo;
        v1hi ^= v2hi;
        // rotl(v2, 32)
        const v2lo_ = v2lo;
        const v2hi_ = v2hi;
        v2lo = v2hi_;
        v2hi = v2lo_;
    }
}

//////////////


// Parts of this code are based on Lucene, which is licensed under the
// Apache/2.0 license.
// More information found here:
// https://fossies.org/linux/lucene/lucene/core/src/java/org/apache/lucene/util/automaton/
//   LevenshteinAutomata.java
class ParametricDescription {
    /**
     * @param {number} w
     * @param {number} n
     * @param {Int32Array} minErrors
     */
    constructor(w, n, minErrors) {
        this.w = w;
        this.n = n;
        this.minErrors = minErrors;
    }
    /**
     * @param {number} absState
     * @returns {boolean}
     */
    isAccept(absState) {
        const state = Math.floor(absState / (this.w + 1));
        const offset = absState % (this.w + 1);
        return this.w - offset + this.minErrors[state] <= this.n;
    }
    /**
     * @param {number} absState
     * @returns {number}
     */
    getPosition(absState) {
        return absState % (this.w + 1);
    }
    /**
     * @param {Uint8Array} name
     * @param {number} charCode
     * @param {number} pos
     * @param {number} end
     * @returns {number}
     */
    getVector(name, charCode, pos, end) {
        let vector = 0;
        for (let i = pos; i < end; i += 1) {
            vector = vector << 1;
            if (name[i] === charCode) {
                vector |= 1;
            }
        }
        return vector;
    }
    /**
     * @param {Int32Array} data
     * @param {number} index
     * @param {number} bitsPerValue
     * @returns {number}
     */
    unpack(data, index, bitsPerValue) {
        const bitLoc = (bitsPerValue * index);
        const dataLoc = bitLoc >> 5;
        const bitStart = bitLoc & 31;
        if (bitStart + bitsPerValue <= 32) {
            // not split
            return ((data[dataLoc] >> bitStart) & this.MASKS[bitsPerValue - 1]);
        } else {
            // split
            const part = 32 - bitStart;
            return ~~(((data[dataLoc] >> bitStart) & this.MASKS[part - 1]) +
                ((data[1 + dataLoc] & this.MASKS[bitsPerValue - part - 1]) << part));
        }
    }
}
ParametricDescription.prototype.MASKS = new Int32Array([
    0x1, 0x3, 0x7, 0xF,
    0x1F, 0x3F, 0x7F, 0xFF,
    0x1FF, 0x3F, 0x7FF, 0xFFF,
    0x1FFF, 0x3FFF, 0x7FFF, 0xFFFF,
    0x1FFFF, 0x3FFFF, 0x7FFFF, 0xFFFFF,
    0x1FFFFF, 0x3FFFFF, 0x7FFFFF, 0xFFFFFF,
    0x1FFFFFF, 0x3FFFFFF, 0x7FFFFFF, 0xFFFFFFF,
    0x1FFFFFFF, 0x3FFFFFFF, 0x7FFFFFFF, 0xFFFFFFFF,
]);

// The following code was generated with the moman/finenight pkg
// This package is available under the MIT License, see NOTICE.txt
// for more details.
// This class is auto-generated, Please do not modify it directly.
// You should modify the https://gitlab.com/notriddle/createAutomata.py instead.
// The following code was generated with the moman/finenight pkg
// This package is available under the MIT License, see NOTICE.txt
// for more details.
// This class is auto-generated, Please do not modify it directly.
// You should modify https://gitlab.com/notriddle/moman-rustdoc instead.

class Lev2TParametricDescription extends ParametricDescription {
    /**
     * @param {number} absState
     * @param {number} position
     * @param {number} vector
     * @returns {number}
    */
    transition(absState, position, vector) {
        let state = Math.floor(absState / (this.w + 1));
        let offset = absState % (this.w + 1);

        if (position === this.w) {
            if (state < 3) {
                const loc = Math.imul(vector, 3) + state;
                offset += this.unpack(this.offsetIncrs0, loc, 1);
                state = this.unpack(this.toStates0, loc, 2) - 1;
            }
        } else if (position === this.w - 1) {
            if (state < 5) {
                const loc = Math.imul(vector, 5) + state;
                offset += this.unpack(this.offsetIncrs1, loc, 1);
                state = this.unpack(this.toStates1, loc, 3) - 1;
            }
        } else if (position === this.w - 2) {
            if (state < 13) {
                const loc = Math.imul(vector, 13) + state;
                offset += this.unpack(this.offsetIncrs2, loc, 2);
                state = this.unpack(this.toStates2, loc, 4) - 1;
            }
        } else if (position === this.w - 3) {
            if (state < 28) {
                const loc = Math.imul(vector, 28) + state;
                offset += this.unpack(this.offsetIncrs3, loc, 2);
                state = this.unpack(this.toStates3, loc, 5) - 1;
            }
        } else if (position === this.w - 4) {
            if (state < 45) {
                const loc = Math.imul(vector, 45) + state;
                offset += this.unpack(this.offsetIncrs4, loc, 3);
                state = this.unpack(this.toStates4, loc, 6) - 1;
            }
        } else {
            // eslint-disable-next-line no-lonely-if
            if (state < 45) {
                const loc = Math.imul(vector, 45) + state;
                offset += this.unpack(this.offsetIncrs5, loc, 3);
                state = this.unpack(this.toStates5, loc, 6) - 1;
            }
        }

        if (state === -1) {
            // null state
            return -1;
        } else {
            // translate back to abs
            return Math.imul(state, this.w + 1) + offset;
        }
    }

    // state map
    //   0 -> [(0, 0)]
    //   1 -> [(0, 1)]
    //   2 -> [(0, 2)]
    //   3 -> [(0, 1), (1, 1)]
    //   4 -> [(0, 2), (1, 2)]
    //   5 -> [(0, 1), (1, 1), (2, 1)]
    //   6 -> [(0, 2), (1, 2), (2, 2)]
    //   7 -> [(0, 1), (2, 1)]
    //   8 -> [(0, 1), (2, 2)]
    //   9 -> [(0, 2), (2, 1)]
    //   10 -> [(0, 2), (2, 2)]
    //   11 -> [t(0, 1), (0, 1), (1, 1), (2, 1)]
    //   12 -> [t(0, 2), (0, 2), (1, 2), (2, 2)]
    //   13 -> [(0, 2), (1, 2), (2, 2), (3, 2)]
    //   14 -> [(0, 1), (1, 1), (3, 2)]
    //   15 -> [(0, 1), (2, 2), (3, 2)]
    //   16 -> [(0, 1), (3, 2)]
    //   17 -> [(0, 1), t(1, 2), (2, 2), (3, 2)]
    //   18 -> [(0, 2), (1, 2), (3, 1)]
    //   19 -> [(0, 2), (1, 2), (3, 2)]
    //   20 -> [(0, 2), (1, 2), t(1, 2), (2, 2), (3, 2)]
    //   21 -> [(0, 2), (2, 1), (3, 1)]
    //   22 -> [(0, 2), (2, 2), (3, 2)]
    //   23 -> [(0, 2), (3, 1)]
    //   24 -> [(0, 2), (3, 2)]
    //   25 -> [(0, 2), t(1, 2), (1, 2), (2, 2), (3, 2)]
    //   26 -> [t(0, 2), (0, 2), (1, 2), (2, 2), (3, 2)]
    //   27 -> [t(0, 2), (0, 2), (1, 2), (3, 1)]
    //   28 -> [(0, 2), (1, 2), (2, 2), (3, 2), (4, 2)]
    //   29 -> [(0, 2), (1, 2), (2, 2), (4, 2)]
    //   30 -> [(0, 2), (1, 2), (2, 2), t(2, 2), (3, 2), (4, 2)]
    //   31 -> [(0, 2), (1, 2), (3, 2), (4, 2)]
    //   32 -> [(0, 2), (1, 2), (4, 2)]
    //   33 -> [(0, 2), (1, 2), t(1, 2), (2, 2), (3, 2), (4, 2)]
    //   34 -> [(0, 2), (1, 2), t(2, 2), (2, 2), (3, 2), (4, 2)]
    //   35 -> [(0, 2), (2, 1), (4, 2)]
    //   36 -> [(0, 2), (2, 2), (3, 2), (4, 2)]
    //   37 -> [(0, 2), (2, 2), (4, 2)]
    //   38 -> [(0, 2), (3, 2), (4, 2)]
    //   39 -> [(0, 2), (4, 2)]
    //   40 -> [(0, 2), t(1, 2), (1, 2), (2, 2), (3, 2), (4, 2)]
    //   41 -> [(0, 2), t(2, 2), (2, 2), (3, 2), (4, 2)]
    //   42 -> [t(0, 2), (0, 2), (1, 2), (2, 2), (3, 2), (4, 2)]
    //   43 -> [t(0, 2), (0, 2), (1, 2), (2, 2), (4, 2)]
    //   44 -> [t(0, 2), (0, 2), (1, 2), (2, 2), t(2, 2), (3, 2), (4, 2)]


    /** @param {number} w - length of word being checked */
    constructor(w) {
        super(w, 2, new Int32Array([
            0,1,2,0,1,-1,0,-1,0,-1,0,-1,0,-1,-1,-1,-1,-1,-2,-1,-1,-2,-1,-2,
            -1,-1,-1,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,
        ]));
    }
}

Lev2TParametricDescription.prototype.toStates0 = /*2 bits per value */ new Int32Array([
    0xe,
]);
Lev2TParametricDescription.prototype.offsetIncrs0 = /*1 bits per value */ new Int32Array([
    0x0,
]);

Lev2TParametricDescription.prototype.toStates1 = /*3 bits per value */ new Int32Array([
    0x1a688a2c,
]);
Lev2TParametricDescription.prototype.offsetIncrs1 = /*1 bits per value */ new Int32Array([
    0x3e0,
]);

Lev2TParametricDescription.prototype.toStates2 = /*4 bits per value */ new Int32Array([
    0x70707054,0xdc07035,0x3dd3a3a,0x2323213a,
    0x15435223,0x22545432,0x5435,
]);
Lev2TParametricDescription.prototype.offsetIncrs2 = /*2 bits per value */ new Int32Array([
    0x80000,0x55582088,0x55555555,0x55,
]);

Lev2TParametricDescription.prototype.toStates3 = /*5 bits per value */ new Int32Array([
    0x1c0380a4,0x700a570,0xca529c0,0x180a00,
    0xa80af180,0xc5498e60,0x5a546398,0x8c4300e8,
    0xac18c601,0xd8d43501,0x863500ad,0x51976d6a,
    0x8ca0180a,0xc3501ac2,0xb0c5be16,0x76dda8a5,
    0x18c4519,0xc41294a,0xe248d231,0x1086520c,
    0xce31ac42,0x13946358,0x2d0348c4,0x6732d494,
    0x1ad224a5,0xd635ad4b,0x520c4139,0xce24948,
    0x22110a52,0x58ce729d,0xc41394e3,0x941cc520,
    0x90e732d4,0x4729d224,0x39ce35ad,
]);
Lev2TParametricDescription.prototype.offsetIncrs3 = /*2 bits per value */ new Int32Array([
    0x80000,0xc0c830,0x300f3c30,0x2200fcff,
    0xcaa00a08,0x3c2200a8,0xa8fea00a,0x55555555,
    0x55555555,0x55555555,0x55555555,0x55555555,
    0x55555555,0x55555555,
]);

Lev2TParametricDescription.prototype.toStates4 = /*6 bits per value */ new Int32Array([
    0x801c0144,0x1453803,0x14700038,0xc0005145,
    0x1401,0x14,0x140000,0x0,
    0x510000,0x6301f007,0x301f00d1,0xa186178,
    0xc20ca0c3,0xc20c30,0xc30030c,0xc00c00cd,
    0xf0c00c30,0x4c054014,0xc30944c3,0x55150c34,
    0x8300550,0x430c0143,0x50c31,0xc30850c,
    0xc3143000,0x50053c50,0x5130d301,0x850d30c2,
    0x30a08608,0xc214414,0x43142145,0x21450031,
    0x1400c314,0x4c143145,0x32832803,0x28014d6c,
    0xcd34a0c3,0x1c50c76,0x1c314014,0x430c30c3,
    0x1431,0xc300500,0xca00d303,0xd36d0e40,
    0x90b0e400,0xcb2abb2c,0x70c20ca1,0x2c32ca2c,
    0xcd2c70cb,0x31c00c00,0x34c2c32c,0x5583280,
    0x558309b7,0x6cd6ca14,0x430850c7,0x51c51401,
    0x1430c714,0xc3087,0x71451450,0xca00d30,
    0xc26dc156,0xb9071560,0x1cb2abb2,0xc70c2144,
    0xb1c51ca1,0x1421c70c,0xc51c00c3,0x30811c51,
    0x24324308,0xc51031c2,0x70820820,0x5c33830d,
    0xc33850c3,0x30c30c30,0xc30c31c,0x451450c3,
    0x20c20c20,0xda0920d,0x5145914f,0x36596114,
    0x51965865,0xd9643653,0x365a6590,0x51964364,
    0x43081505,0x920b2032,0x2c718b28,0xd7242249,
    0x35cb28b0,0x2cb3872c,0x972c30d7,0xb0c32cb2,
    0x4e1c75c,0xc80c90c2,0x62ca2482,0x4504171c,
    0xd65d9610,0x33976585,0xd95cb5d,0x4b5ca5d7,
    0x73975c36,0x10308138,0xc2245105,0x41451031,
    0x14e24208,0xc35c3387,0x51453851,0x1c51c514,
    0xc70c30c3,0x20451450,0x14f1440c,0x4f0da092,
    0x4513d41,0x6533944d,0x1350e658,0xe1545055,
    0x64365a50,0x5519383,0x51030815,0x28920718,
    0x441c718b,0x714e2422,0x1c35cb28,0x4e1c7387,
    0xb28e1c51,0x5c70c32c,0xc204e1c7,0x81c61440,
    0x1c62ca24,0xd04503ce,0x85d63944,0x39338e65,
    0x8e154387,0x364b5ca3,0x38739738,
]);
Lev2TParametricDescription.prototype.offsetIncrs4 = /*3 bits per value */ new Int32Array([
    0x10000000,0xc00000,0x60061,0x400,
    0x0,0x80010008,0x249248a4,0x8229048,
    0x2092,0x6c3603,0xb61b6c30,0x6db6036d,
    0xdb6c0,0x361b0180,0x91b72000,0xdb11b71b,
    0x6db6236,0x1008200,0x12480012,0x24924906,
    0x48200049,0x80410002,0x24000900,0x4924a489,
    0x10822492,0x20800125,0x48360,0x9241b692,
    0x6da4924,0x40009268,0x241b010,0x291b4900,
    0x6d249249,0x49493423,0x92492492,0x24924924,
    0x49249249,0x92492492,0x24924924,0x49249249,
    0x92492492,0x24924924,0x49249249,0x92492492,
    0x24924924,0x49249249,0x92492492,0x24924924,
    0x49249249,0x92492492,0x24924924,0x49249249,
    0x92492492,0x24924924,0x49249249,0x92492492,
    0x24924924,0x49249249,0x92492492,0x24924924,
    0x49249249,0x92492492,0x24924924,0x49249249,
    0x92492492,0x24924924,0x49249249,0x2492,
]);

Lev2TParametricDescription.prototype.toStates5 = /*6 bits per value */ new Int32Array([
    0x801c0144,0x1453803,0x14700038,0xc0005145,
    0x1401,0x14,0x140000,0x0,
    0x510000,0x4e00e007,0xe0051,0x3451451c,
    0xd015000,0x30cd0000,0xc30c30c,0xc30c30d4,
    0x40c30c30,0x7c01c014,0xc03458c0,0x185e0c07,
    0x2830c286,0x830c3083,0xc30030,0x33430c,
    0x30c3003,0x70051030,0x16301f00,0x8301f00d,
    0x30a18617,0xc20ca0c,0x431420c3,0xb1450c51,
    0x14314315,0x4f143145,0x34c05401,0x4c30944c,
    0x55150c3,0x30830055,0x1430c014,0xc00050c3,
    0xc30850,0xc314300,0x150053c5,0x25130d30,
    0x5430d30c,0xc0354154,0x300d0c90,0x1cb2cd0c,
    0xc91cb0c3,0x72c30cb2,0x14f1cb2c,0xc34c0540,
    0x34c30944,0x82182214,0x851050c2,0x50851430,
    0x1400c50c,0x30c5085,0x50c51450,0x150053c,
    0xc25130d3,0x8850d30,0x1430a086,0x450c2144,
    0x51cb1c21,0x1c91c70c,0xc71c314b,0x34c1cb1,
    0x6c328328,0xc328014d,0x76cd34a0,0x1401c50c,
    0xc31c3140,0x31430c30,0x14,0x30c3005,
    0xa0ca00d3,0x535b0c,0x4d2830ca,0x514369b3,
    0xc500d01,0x5965965a,0x30d46546,0x6435030c,
    0x8034c659,0xdb439032,0x2c390034,0xcaaecb24,
    0x30832872,0xcb28b1c,0x4b1c32cb,0x70030033,
    0x30b0cb0c,0xe40ca00d,0x400d36d0,0xb2c90b0e,
    0xca1cb2ab,0xa2c70c20,0x6575d95c,0x4315b5ce,
    0x95c53831,0x28034c5d,0x9b705583,0xa1455830,
    0xc76cd6c,0x40143085,0x71451c51,0x871430c,
    0x450000c3,0xd3071451,0x1560ca00,0x560c26dc,
    0xb35b2851,0xc914369,0x1a14500d,0x46593945,
    0xcb2c939,0x94507503,0x328034c3,0x9b70558,
    0xe41c5583,0x72caaeca,0x1c308510,0xc7147287,
    0x50871c32,0x1470030c,0xd307147,0xc1560ca0,
    0x1560c26d,0xabb2b907,0x21441cb2,0x38a1c70c,
    0x8e657394,0x314b1c93,0x39438738,0x43083081,
    0x31c22432,0x820c510,0x830d7082,0x50c35c33,
    0xc30c338,0xc31c30c3,0x50c30c30,0xc204514,
    0x890c90c2,0x31440c70,0xa8208208,0xea0df0c3,
    0x8a231430,0xa28a28a2,0x28a28a1e,0x1861868a,
    0x48308308,0xc3682483,0x14516453,0x4d965845,
    0xd4659619,0x36590d94,0xd969964,0x546590d9,
    0x20c20541,0x920d20c,0x5914f0da,0x96114514,
    0x65865365,0xe89d3519,0x99e7a279,0x9e89e89e,
    0x81821827,0xb2032430,0x18b28920,0x422492c7,
    0xb28b0d72,0x3872c35c,0xc30d72cb,0x32cb2972,
    0x1c75cb0c,0xc90c204e,0xa2482c80,0x24b1c62c,
    0xc3a89089,0xb0ea2e42,0x9669a31c,0xa4966a28,
    0x59a8a269,0x8175e7a,0xb203243,0x718b2892,
    0x4114105c,0x17597658,0x74ce5d96,0x5c36572d,
    0xd92d7297,0xe1ce5d70,0xc90c204,0xca2482c8,
    0x4171c62,0x5d961045,0x976585d6,0x79669533,
    0x964965a2,0x659689e6,0x308175e7,0x24510510,
    0x451031c2,0xe2420841,0x5c338714,0x453851c3,
    0x51c51451,0xc30c31c,0x451450c7,0x41440c20,
    0xc708914,0x82105144,0xf1c58c90,0x1470ea0d,
    0x61861863,0x8a1e85e8,0x8687a8a2,0x3081861,
    0x24853c51,0x5053c368,0x1341144f,0x96194ce5,
    0x1544d439,0x94385514,0xe0d90d96,0x5415464,
    0x4f1440c2,0xf0da0921,0x4513d414,0x533944d0,
    0x350e6586,0x86082181,0xe89e981d,0x18277689,
    0x10308182,0x89207185,0x41c718b2,0x14e24224,
    0xc35cb287,0xe1c73871,0x28e1c514,0xc70c32cb,
    0x204e1c75,0x1c61440c,0xc62ca248,0x90891071,
    0x2e41c58c,0xa31c70ea,0xe86175e7,0xa269a475,
    0x5e7a57a8,0x51030817,0x28920718,0xf38718b,
    0xe5134114,0x39961758,0xe1ce4ce,0x728e3855,
    0x5ce0d92d,0xc204e1ce,0x81c61440,0x1c62ca24,
    0xd04503ce,0x85d63944,0x75338e65,0x5d86075e,
    0x89e69647,0x75e76576,
]);
Lev2TParametricDescription.prototype.offsetIncrs5 = /*3 bits per value */ new Int32Array([
    0x10000000,0xc00000,0x60061,0x400,
    0x0,0x60000008,0x6b003080,0xdb6ab6db,
    0x2db6,0x800400,0x49245240,0x11482412,
    0x104904,0x40020000,0x92292000,0xa4b25924,
    0x9649658,0xd80c000,0xdb0c001b,0x80db6d86,
    0x6db01b6d,0xc0600003,0x86000d86,0x6db6c36d,
    0xddadb6ed,0x300001b6,0x6c360,0xe37236e4,
    0x46db6236,0xdb6c,0x361b018,0xb91b7200,
    0x6dbb1b71,0x6db763,0x20100820,0x61248001,
    0x92492490,0x24820004,0x8041000,0x92400090,
    0x24924830,0x555b6a49,0x2080012,0x20004804,
    0x49252449,0x84112492,0x4000928,0x240201,
    0x92922490,0x58924924,0x49456,0x120d8082,
    0x6da4800,0x69249249,0x249a01b,0x6c04100,
    0x6d240009,0x92492483,0x24d5adb4,0x60208001,
    0x92000483,0x24925236,0x6846da49,0x10400092,
    0x241b0,0x49291b49,0x636d2492,0x92494935,
    0x24924924,0x49249249,0x92492492,0x24924924,
    0x49249249,0x92492492,0x24924924,0x49249249,
    0x92492492,0x24924924,0x49249249,0x92492492,
    0x24924924,0x49249249,0x92492492,0x24924924,
    0x49249249,0x92492492,0x24924924,0x49249249,
    0x92492492,0x24924924,0x49249249,0x92492492,
    0x24924924,0x49249249,0x92492492,0x24924924,
    0x49249249,0x92492492,0x24924924,0x49249249,
    0x92492492,0x24924924,0x49249249,0x92492492,
    0x24924924,0x49249249,0x92492492,0x24924924,
    0x49249249,0x92492492,0x24924924,0x49249249,
    0x92492492,0x24924924,0x49249249,0x92492492,
    0x24924924,0x49249249,0x92492492,0x24924924,
    0x49249249,0x92492492,0x24924924,0x49249249,
    0x92492492,0x24924924,0x49249249,0x92492492,
    0x24924924,0x49249249,0x92492492,0x24924924,
    0x49249249,0x92492492,0x24924924,
]);

class Lev1TParametricDescription extends ParametricDescription {
    /**
     * @param {number} absState
     * @param {number} position
     * @param {number} vector
     * @returns {number}
    */
    transition(absState, position, vector) {
        let state = Math.floor(absState / (this.w + 1));
        let offset = absState % (this.w + 1);

        if (position === this.w) {
            if (state < 2) {
                const loc = Math.imul(vector, 2) + state;
                offset += this.unpack(this.offsetIncrs0, loc, 1);
                state = this.unpack(this.toStates0, loc, 2) - 1;
            }
        } else if (position === this.w - 1) {
            if (state < 3) {
                const loc = Math.imul(vector, 3) + state;
                offset += this.unpack(this.offsetIncrs1, loc, 1);
                state = this.unpack(this.toStates1, loc, 2) - 1;
            }
        } else if (position === this.w - 2) {
            if (state < 6) {
                const loc = Math.imul(vector, 6) + state;
                offset += this.unpack(this.offsetIncrs2, loc, 2);
                state = this.unpack(this.toStates2, loc, 3) - 1;
            }
        } else {
            // eslint-disable-next-line no-lonely-if
            if (state < 6) {
                const loc = Math.imul(vector, 6) + state;
                offset += this.unpack(this.offsetIncrs3, loc, 2);
                state = this.unpack(this.toStates3, loc, 3) - 1;
            }
        }

        if (state === -1) {
            // null state
            return -1;
        } else {
            // translate back to abs
            return Math.imul(state, this.w + 1) + offset;
        }
    }

    // state map
    //   0 -> [(0, 0)]
    //   1 -> [(0, 1)]
    //   2 -> [(0, 1), (1, 1)]
    //   3 -> [(0, 1), (1, 1), (2, 1)]
    //   4 -> [(0, 1), (2, 1)]
    //   5 -> [t(0, 1), (0, 1), (1, 1), (2, 1)]


    /** @param {number} w - length of word being checked */
    constructor(w) {
        super(w, 1, new Int32Array([0,1,0,-1,-1,-1]));
    }
}

Lev1TParametricDescription.prototype.toStates0 = /*2 bits per value */ new Int32Array([
    0x2,
]);
Lev1TParametricDescription.prototype.offsetIncrs0 = /*1 bits per value */ new Int32Array([
    0x0,
]);

Lev1TParametricDescription.prototype.toStates1 = /*2 bits per value */ new Int32Array([
    0xa43,
]);
Lev1TParametricDescription.prototype.offsetIncrs1 = /*1 bits per value */ new Int32Array([
    0x38,
]);

Lev1TParametricDescription.prototype.toStates2 = /*3 bits per value */ new Int32Array([
    0x12180003,0xb45a4914,0x69,
]);
Lev1TParametricDescription.prototype.offsetIncrs2 = /*2 bits per value */ new Int32Array([
    0x558a0000,0x5555,
]);

Lev1TParametricDescription.prototype.toStates3 = /*3 bits per value */ new Int32Array([
    0x900c0003,0xa1904864,0x45a49169,0x5a6d196a,
    0x9634,
]);
Lev1TParametricDescription.prototype.offsetIncrs3 = /*2 bits per value */ new Int32Array([
    0xa0fc0000,0x5555ba08,0x55555555,
]);
