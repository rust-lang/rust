//@ revisions: legacy legacy-ok hashed hashed-ok
//@ [legacy] compile-flags: -Csymbol-mangling-version=legacy
//@ [legacy-ok] check-pass
//@ [legacy-ok] compile-flags: -Zunstable-options -Csymbol-mangling-version=legacy
//@ [hashed] compile-flags: -Csymbol-mangling-version=hashed
//@ [hashed-ok] check-pass
//@ [hashed-ok] compile-flags: -Zunstable-options -Csymbol-mangling-version=hashed

fn main() {}
