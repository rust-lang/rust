//@ compile-flags:-g

// FIXME(jieyouxu): triple check if this works in CI
//@ min-cdb-version: 10.0.26100.2161

// === GDB TESTS ===================================================================================

// gdb-command: run

// gdb-command: print _ref
// gdb-check: $1 = (*mut ()) 0x[...]

// gdb-command: print _ptr
// gdb-check: $2 = (*mut ()) 0x[...]

// gdb-command: print _local
// gdb-check: $3 = ()

// gdb-command: print _field
// gdb-check: $4 = unit_type::_TypeContainingUnitField {_a: 123, _unit: (), _b: 456}

// Check that we can cast "void pointers" to their actual type in the debugger
// gdb-command: print /x *(_ptr as *const u64)
// gdb-check: $5 = 0x1122334455667788

// === CDB TESTS ===================================================================================

// cdb-command: g
// cdb-check: Breakpoint 0 hit

// cdb-command: dx _ref
// cdb-check: _ref             : 0x[...] [Type: tuple$<> *]

// cdb-command: dx _ptr
// cdb-check: _ptr             : 0x[...] [Type: tuple$<> *]

// cdb-command: dx _local
// cdb-check: _local           [Type: tuple$<>]

// cdb-command: dx _field,d
// cdb-check: _field,d         [Type: unit_type::_TypeContainingUnitField]
// cdb-check:     [+0x[...]] _a               : 123 [Type: unsigned int]
// cdb-check:     [+0x[...]] _unit            [Type: tuple$<>]
// cdb-check:     [+0x[...]] _b               : 456 [Type: unsigned __int64]

// Check that we can cast "void pointers" to their actual type in the debugger
// cdb-command: dx ((__int64 *)_ptr),x
// cdb-check: ((__int64 *)_ptr),x : 0x[...] : 0x1122334455667788 [Type: __int64 *]
// cdb-check:     0x1122334455667788 [Type: __int64]

struct _TypeContainingUnitField {
    _a: u32,
    _unit: (),
    _b: u64,
}

fn foo(_ref: &(), _ptr: *const ()) {
    let _local = ();
    let _field = _TypeContainingUnitField { _a: 123, _unit: (), _b: 456 };

    zzz(); // #break
}

fn main() {
    let pointee = 0x1122_3344_5566_7788i64;

    foo(&(), &pointee as *const i64 as *const ());
}

#[inline(never)]
fn zzz() {}
