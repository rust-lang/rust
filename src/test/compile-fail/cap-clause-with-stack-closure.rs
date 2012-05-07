fn foo(_f: fn()) {}
fn bar(_f: @int) {}

fn main() {
    let x = @3;
    foo {|| bar(x); }

    let x = @3;
    foo {|copy x| bar(x); } //! ERROR cannot capture values explicitly with a block closure

    let x = @3;
    foo {|move x| bar(x); } //! ERROR cannot capture values explicitly with a block closure
}

