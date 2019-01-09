//compile-flags: -Z borrowck=compare

fn foo(_x: u32) {
    _x = 4;
    //~^ ERROR cannot assign to immutable argument `_x` (Mir)
    //~^^ ERROR cannot assign twice to immutable variable `_x` (Ast)
}

fn main() {}

