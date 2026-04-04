import gdb
import gdb.printing
import re

import gdb_providers as providers
from rust_types import RustType, classify_struct, classify_union


_gdb_version_matched = re.search("([0-9]+)\\.([0-9]+)", gdb.VERSION)
gdb_version = (
    [int(num) for num in _gdb_version_matched.groups()] if _gdb_version_matched else []
)


def register_printers(objfile):
    objfile.pretty_printers.append(printer)


# BACKCOMPAT: rust 1.35
def is_hashbrown_hashmap(hash_map):
    return len(hash_map.type.fields()) == 1


def classify_rust_type(type):
    type_class = type.code
    if type_class == gdb.TYPE_CODE_STRUCT:
        return classify_struct(type.tag, type.fields())
    if type_class == gdb.TYPE_CODE_UNION:
        return classify_union(type.fields())

    return RustType.OTHER


def check_enum_discriminant(valobj):
    content = valobj[valobj.type.fields()[0]]
    fields = content.type.fields()
    if len(fields) > 1:
        discriminant = int(content[fields[0]]) + 1
        if discriminant > len(fields):
            # invalid discriminant
            return False
    return True


class PrintByRustType(gdb.printing.SubPrettyPrinter):
    def __init__(self, name, provider):
        super(PrintByRustType, self).__init__(name)
        self.provider = provider

    def __call__(self, val):
        if self.enabled:
            return self.provider(val)
        return None


class RustPrettyPrinter(gdb.printing.PrettyPrinter):
    def __init__(self, name):
        super(RustPrettyPrinter, self).__init__(name, [])
        self.type_map = {}

    def add(self, rust_type, provider):
        # Just use the rust_type as the name.
        printer = PrintByRustType(rust_type, provider)
        self.type_map[rust_type] = printer
        self.subprinters.append(printer)

    def __call__(self, valobj):
        rust_type = classify_rust_type(valobj.type)
        if rust_type in self.type_map:
            return self.type_map[rust_type](valobj)
        return None


printer = RustPrettyPrinter("rust")
printer.add(RustType.STD_STRING, providers.StdStringProvider)
printer.add(RustType.STD_OS_STRING, providers.StdOsStringProvider)
printer.add(RustType.STD_STR, providers.StdStrProvider)
printer.add(RustType.STD_SLICE, providers.StdSliceProvider)
printer.add(RustType.STD_VEC, providers.StdVecProvider)
printer.add(RustType.STD_VEC_DEQUE, providers.StdVecDequeProvider)
printer.add(RustType.STD_BTREE_SET, providers.StdBTreeSetProvider)
printer.add(RustType.STD_BTREE_MAP, providers.StdBTreeMapProvider)
printer.add(RustType.STD_HASH_MAP, providers.StdHashMapProvider)
printer.add(RustType.STD_HASH_SET, providers.StdHashMapProvider)
printer.add(RustType.STD_RC, providers.StdRcProvider)
printer.add(
    RustType.STD_ARC, lambda valobj: providers.StdRcProvider(valobj, is_atomic=True)
)

printer.add(RustType.STD_CELL, providers.StdCellProvider)
printer.add(RustType.STD_REF, providers.StdRefProvider)
printer.add(RustType.STD_REF_MUT, providers.StdRefProvider)
printer.add(RustType.STD_REF_CELL, providers.StdRefCellProvider)
printer.add(RustType.STD_NONZERO_NUMBER, providers.StdNonZeroNumberProvider)
