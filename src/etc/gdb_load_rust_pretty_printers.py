import gdb
import gdb_lookup
gdb_lookup.register_printers(gdb.current_objfile())
