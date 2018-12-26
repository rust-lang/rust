// compile-flags: -Z borrowck=compare

pub fn main(){
    let maybe = Some(vec![true, true]);

    loop {
        if let Some(thing) = maybe {
        }
        //~^^ ERROR use of partially moved value: `maybe` (Ast) [E0382]
        //~| ERROR use of moved value: `(maybe as std::prelude::v1::Some).0` (Ast) [E0382]
        //~| ERROR use of moved value (Mir) [E0382]
    }
}
