#[link(name = "crateresolve5",
       vers = "0.1")];

#[crate_type = "lib"];

fn structural() -> { name: ~str, val: int } {
    { name: ~"crateresolve5", val: 10 }
}

enum e {
    e_val
}

fn nominal() -> e { e_val }

fn f() -> int { 10 }
