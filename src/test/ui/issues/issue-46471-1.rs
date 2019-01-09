// compile-flags: -Z borrowck=compare

fn main() {
    let y = {
        let mut z = 0;
        &mut z
    };
    //~^^ ERROR `z` does not live long enough (Ast) [E0597]
    //~| ERROR `z` does not live long enough (Mir) [E0597]
    println!("{}", y);
}
