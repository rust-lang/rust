// compile-flags: --crate-type=lib

// Preserve semicolons that disambiguate unops

fn f() { }

fn block_semi() -> isize { { f() }; -1 }

fn block_nosemi() -> isize { ({ 0 }) - 1 }

fn if_semi() -> isize { if true { f() } else { f() }; -1 }

fn if_nosemi() -> isize { (if true { 0 } else { 0 }) - 1 }

fn alt_semi() -> isize { match true { true => { f() } _ => { } }; -1 }

fn alt_no_semi() -> isize { (match true { true => { 0 } _ => { 1 } }) - 1 }

fn stmt() { { f() }; -1; }
