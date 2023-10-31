export type NotNull<T> = T extends null ? never : T;

export type Nullable<T> = T | null;

function isNotNull<T>(input: Nullable<T>): input is NotNull<T> {
    return input !== null;
}

function expectNotNull<T>(input: Nullable<T>, msg: string): NotNull<T> {
    if (isNotNull(input)) {
        return input;
    }

    throw new TypeError(msg);
}

export function unwrapNullable<T>(input: Nullable<T>): NotNull<T> {
    return expectNotNull(input, `unwrapping \`null\``);
}
