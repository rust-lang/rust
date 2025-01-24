from sys import version_info

import gdb

if version_info[0] >= 3:
    xrange = range

ZERO_FIELD = "__0"
FIRST_FIELD = "__1"


def unwrap_unique_or_non_null(unique_or_nonnull):
    # BACKCOMPAT: rust 1.32
    # https://github.com/rust-lang/rust/commit/7a0911528058e87d22ea305695f4047572c5e067
    # BACKCOMPAT: rust 1.60
    # https://github.com/rust-lang/rust/commit/2a91eeac1a2d27dd3de1bf55515d765da20fd86f
    ptr = unique_or_nonnull["pointer"]
    return ptr if ptr.type.code == gdb.TYPE_CODE_PTR else ptr[ptr.type.fields()[0]]


# GDB 14 has a tag class that indicates that extension methods are ok
# to call.  Use of this tag only requires that printers hide local
# attributes and methods by prefixing them with "_".
if hasattr(gdb, "ValuePrinter"):
    printer_base = gdb.ValuePrinter
else:
    printer_base = object


class EnumProvider(printer_base):
    def __init__(self, valobj):
        content = valobj[valobj.type.fields()[0]]
        fields = content.type.fields()
        self._empty = len(fields) == 0
        if not self._empty:
            if len(fields) == 1:
                discriminant = 0
            else:
                discriminant = int(content[fields[0]]) + 1
            self._active_variant = content[fields[discriminant]]
            self._name = fields[discriminant].name
            self._full_name = "{}::{}".format(valobj.type.name, self._name)
        else:
            self._full_name = valobj.type.name

    def to_string(self):
        return self._full_name

    def children(self):
        if not self._empty:
            yield self._name, self._active_variant


class StdStringProvider(printer_base):
    def __init__(self, valobj):
        self._valobj = valobj
        vec = valobj["vec"]
        self._length = int(vec["len"])
        self._data_ptr = unwrap_unique_or_non_null(vec["buf"]["inner"]["ptr"])

    def to_string(self):
        return self._data_ptr.lazy_string(encoding="utf-8", length=self._length)

    @staticmethod
    def display_hint():
        return "string"


class StdOsStringProvider(printer_base):
    def __init__(self, valobj):
        self._valobj = valobj
        buf = self._valobj["inner"]["inner"]
        is_windows = "Wtf8Buf" in buf.type.name
        vec = buf[ZERO_FIELD] if is_windows else buf

        self._length = int(vec["len"])
        self._data_ptr = unwrap_unique_or_non_null(vec["buf"]["inner"]["ptr"])

    def to_string(self):
        return self._data_ptr.lazy_string(encoding="utf-8", length=self._length)

    def display_hint(self):
        return "string"


class StdStrProvider(printer_base):
    def __init__(self, valobj):
        self._valobj = valobj
        self._length = int(valobj["length"])
        self._data_ptr = valobj["data_ptr"]

    def to_string(self):
        return self._data_ptr.lazy_string(encoding="utf-8", length=self._length)

    @staticmethod
    def display_hint():
        return "string"


def _enumerate_array_elements(element_ptrs):
    for i, element_ptr in enumerate(element_ptrs):
        key = "[{}]".format(i)
        element = element_ptr.dereference()

        try:
            # rust-lang/rust#64343: passing deref expr to `str` allows
            # catching exception on garbage pointer
            str(element)
        except RuntimeError:
            yield key, "inaccessible"

            break

        yield key, element


class StdSliceProvider(printer_base):
    def __init__(self, valobj):
        self._valobj = valobj
        self._length = int(valobj["length"])
        self._data_ptr = valobj["data_ptr"]

    def to_string(self):
        return "{}(size={})".format(self._valobj.type, self._length)

    def children(self):
        return _enumerate_array_elements(
            self._data_ptr + index for index in xrange(self._length)
        )

    @staticmethod
    def display_hint():
        return "array"


class StdVecProvider(printer_base):
    def __init__(self, valobj):
        self._valobj = valobj
        self._length = int(valobj["len"])
        self._data_ptr = unwrap_unique_or_non_null(valobj["buf"]["inner"]["ptr"])
        ptr_ty = gdb.Type.pointer(valobj.type.template_argument(0))
        self._data_ptr = self._data_ptr.reinterpret_cast(ptr_ty)

    def to_string(self):
        return "Vec(size={})".format(self._length)

    def children(self):
        return _enumerate_array_elements(
            self._data_ptr + index for index in xrange(self._length)
        )

    @staticmethod
    def display_hint():
        return "array"


class StdVecDequeProvider(printer_base):
    def __init__(self, valobj):
        self._valobj = valobj
        self._head = int(valobj["head"])
        self._size = int(valobj["len"])
        # BACKCOMPAT: rust 1.75
        cap = valobj["buf"]["inner"]["cap"]
        if cap.type.code != gdb.TYPE_CODE_INT:
            cap = cap[ZERO_FIELD]
        self._cap = int(cap)
        self._data_ptr = unwrap_unique_or_non_null(valobj["buf"]["inner"]["ptr"])
        ptr_ty = gdb.Type.pointer(valobj.type.template_argument(0))
        self._data_ptr = self._data_ptr.reinterpret_cast(ptr_ty)

    def to_string(self):
        return "VecDeque(size={})".format(self._size)

    def children(self):
        return _enumerate_array_elements(
            (self._data_ptr + ((self._head + index) % self._cap))
            for index in xrange(self._size)
        )

    @staticmethod
    def display_hint():
        return "array"


class StdRcProvider(printer_base):
    def __init__(self, valobj, is_atomic=False):
        self._valobj = valobj
        self._is_atomic = is_atomic
        self._ptr = unwrap_unique_or_non_null(valobj["ptr"])
        self._value = self._ptr["data" if is_atomic else "value"]
        self._strong = self._ptr["strong"]["v" if is_atomic else "value"]["value"]
        self._weak = self._ptr["weak"]["v" if is_atomic else "value"]["value"] - 1

    def to_string(self):
        if self._is_atomic:
            return "Arc(strong={}, weak={})".format(int(self._strong), int(self._weak))
        else:
            return "Rc(strong={}, weak={})".format(int(self._strong), int(self._weak))

    def children(self):
        yield "value", self._value
        yield "strong", self._strong
        yield "weak", self._weak


class StdCellProvider(printer_base):
    def __init__(self, valobj):
        self._value = valobj["value"]["value"]

    def to_string(self):
        return "Cell"

    def children(self):
        yield "value", self._value


class StdRefProvider(printer_base):
    def __init__(self, valobj):
        self._value = valobj["value"].dereference()
        self._borrow = valobj["borrow"]["borrow"]["value"]["value"]

    def to_string(self):
        borrow = int(self._borrow)
        if borrow >= 0:
            return "Ref(borrow={})".format(borrow)
        else:
            return "Ref(borrow_mut={})".format(-borrow)

    def children(self):
        yield "*value", self._value
        yield "borrow", self._borrow


class StdRefCellProvider(printer_base):
    def __init__(self, valobj):
        self._value = valobj["value"]["value"]
        self._borrow = valobj["borrow"]["value"]["value"]

    def to_string(self):
        borrow = int(self._borrow)
        if borrow >= 0:
            return "RefCell(borrow={})".format(borrow)
        else:
            return "RefCell(borrow_mut={})".format(-borrow)

    def children(self):
        yield "value", self._value
        yield "borrow", self._borrow


class StdNonZeroNumberProvider(printer_base):
    def __init__(self, valobj):
        fields = valobj.type.fields()
        assert len(fields) == 1
        field = list(fields)[0]

        inner_valobj = valobj[field.name]

        inner_fields = inner_valobj.type.fields()
        assert len(inner_fields) == 1
        inner_field = list(inner_fields)[0]

        self._value = str(inner_valobj[inner_field.name])

    def to_string(self):
        return self._value


# Yields children (in a provider's sense of the word) for a BTreeMap.
def children_of_btree_map(map):
    # Yields each key/value pair in the node and in any child nodes.
    def children_of_node(node_ptr, height):
        def cast_to_internal(node):
            internal_type_name = node.type.target().name.replace(
                "LeafNode", "InternalNode", 1
            )
            internal_type = gdb.lookup_type(internal_type_name)
            return node.cast(internal_type.pointer())

        if node_ptr.type.name.startswith("alloc::collections::btree::node::BoxedNode<"):
            # BACKCOMPAT: rust 1.49
            node_ptr = node_ptr["ptr"]
        node_ptr = unwrap_unique_or_non_null(node_ptr)
        leaf = node_ptr.dereference()
        keys = leaf["keys"]
        vals = leaf["vals"]
        edges = cast_to_internal(node_ptr)["edges"] if height > 0 else None
        length = leaf["len"]

        for i in xrange(0, length + 1):
            if height > 0:
                child_ptr = edges[i]["value"]["value"]
                for child in children_of_node(child_ptr, height - 1):
                    yield child
            if i < length:
                # Avoid "Cannot perform pointer math on incomplete type" on zero-sized arrays.
                key_type_size = keys.type.sizeof
                val_type_size = vals.type.sizeof
                key = (
                    keys[i]["value"]["value"]
                    if key_type_size > 0
                    else gdb.parse_and_eval("()")
                )
                val = (
                    vals[i]["value"]["value"]
                    if val_type_size > 0
                    else gdb.parse_and_eval("()")
                )
                yield key, val

    if map["length"] > 0:
        root = map["root"]
        if root.type.name.startswith("core::option::Option<"):
            root = root.cast(gdb.lookup_type(root.type.name[21:-1]))
        node_ptr = root["node"]
        height = root["height"]
        for child in children_of_node(node_ptr, height):
            yield child


class StdBTreeSetProvider(printer_base):
    def __init__(self, valobj):
        self._valobj = valobj

    def to_string(self):
        return "BTreeSet(size={})".format(self._valobj["map"]["length"])

    def children(self):
        inner_map = self._valobj["map"]
        for i, (child, _) in enumerate(children_of_btree_map(inner_map)):
            yield "[{}]".format(i), child

    @staticmethod
    def display_hint():
        return "array"


class StdBTreeMapProvider(printer_base):
    def __init__(self, valobj):
        self._valobj = valobj

    def to_string(self):
        return "BTreeMap(size={})".format(self._valobj["length"])

    def children(self):
        for i, (key, val) in enumerate(children_of_btree_map(self._valobj)):
            yield "key{}".format(i), key
            yield "val{}".format(i), val

    @staticmethod
    def display_hint():
        return "map"


# BACKCOMPAT: rust 1.35
class StdOldHashMapProvider(printer_base):
    def __init__(self, valobj, show_values=True):
        self._valobj = valobj
        self._show_values = show_values

        self._table = self._valobj["table"]
        self._size = int(self._table["size"])
        self._hashes = self._table["hashes"]
        self._hash_uint_type = self._hashes.type
        self._hash_uint_size = self._hashes.type.sizeof
        self._modulo = 2**self._hash_uint_size
        self._data_ptr = self._hashes[ZERO_FIELD]["pointer"]

        self._capacity_mask = int(self._table["capacity_mask"])
        self._capacity = (self._capacity_mask + 1) % self._modulo

        marker = self._table["marker"].type
        self._pair_type = marker.template_argument(0)
        self._pair_type_size = self._pair_type.sizeof

        self._valid_indices = []
        for idx in range(self._capacity):
            data_ptr = self._data_ptr.cast(self._hash_uint_type.pointer())
            address = data_ptr + idx
            hash_uint = address.dereference()
            hash_ptr = hash_uint[ZERO_FIELD]["pointer"]
            if int(hash_ptr) != 0:
                self._valid_indices.append(idx)

    def to_string(self):
        if self._show_values:
            return "HashMap(size={})".format(self._size)
        else:
            return "HashSet(size={})".format(self._size)

    def children(self):
        start = int(self._data_ptr) & ~1

        hashes = self._hash_uint_size * self._capacity
        align = self._pair_type_size
        len_rounded_up = (
            (
                (((hashes + align) % self._modulo - 1) % self._modulo)
                & ~((align - 1) % self._modulo)
            )
            % self._modulo
            - hashes
        ) % self._modulo

        pairs_offset = hashes + len_rounded_up
        pairs_start = gdb.Value(start + pairs_offset).cast(self._pair_type.pointer())

        for index in range(self._size):
            table_index = self._valid_indices[index]
            idx = table_index & self._capacity_mask
            element = (pairs_start + idx).dereference()
            if self._show_values:
                yield "key{}".format(index), element[ZERO_FIELD]
                yield "val{}".format(index), element[FIRST_FIELD]
            else:
                yield "[{}]".format(index), element[ZERO_FIELD]

    def display_hint(self):
        return "map" if self._show_values else "array"


class StdHashMapProvider(printer_base):
    def __init__(self, valobj, show_values=True):
        self._valobj = valobj
        self._show_values = show_values

        table = self._table()
        table_inner = table["table"]
        capacity = int(table_inner["bucket_mask"]) + 1
        ctrl = table_inner["ctrl"]["pointer"]

        self._size = int(table_inner["items"])
        self._pair_type = table.type.template_argument(0).strip_typedefs()

        self._new_layout = not table_inner.type.has_key("data")
        if self._new_layout:
            self._data_ptr = ctrl.cast(self._pair_type.pointer())
        else:
            self._data_ptr = table_inner["data"]["pointer"]

        self._valid_indices = []
        for idx in range(capacity):
            address = ctrl + idx
            value = address.dereference()
            is_presented = value & 128 == 0
            if is_presented:
                self._valid_indices.append(idx)

    def _table(self):
        if self._show_values:
            hashbrown_hashmap = self._valobj["base"]
        elif self._valobj.type.fields()[0].name == "map":
            # BACKCOMPAT: rust 1.47
            # HashSet wraps std::collections::HashMap, which wraps hashbrown::HashMap
            hashbrown_hashmap = self._valobj["map"]["base"]
        else:
            # HashSet wraps hashbrown::HashSet, which wraps hashbrown::HashMap
            hashbrown_hashmap = self._valobj["base"]["map"]
        return hashbrown_hashmap["table"]

    def to_string(self):
        if self._show_values:
            return "HashMap(size={})".format(self._size)
        else:
            return "HashSet(size={})".format(self._size)

    def children(self):
        pairs_start = self._data_ptr

        for index in range(self._size):
            idx = self._valid_indices[index]
            if self._new_layout:
                idx = -(idx + 1)
            element = (pairs_start + idx).dereference()
            if self._show_values:
                yield "key{}".format(index), element[ZERO_FIELD]
                yield "val{}".format(index), element[FIRST_FIELD]
            else:
                yield "[{}]".format(index), element[ZERO_FIELD]

    def display_hint(self):
        return "map" if self._show_values else "array"
