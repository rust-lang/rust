// Preserve semicolons that disambiguate unops

fn f() { }

fn block_semi() -> int { { f() }; -1 }

fn block_nosemi() -> int { { 0 } - 1 }

fn if_semi() -> int { if true { f() } else { f() }; -1 }

fn if_nosemi() -> int { if true { 0 } else { 0 } - 1 }

fn alt_semi() -> int { alt true { true { f() } }; -1 }

fn alt_no_semi() -> int { alt true { true { 0 } } - 1 }

fn stmt() { { f() }; -1; }
