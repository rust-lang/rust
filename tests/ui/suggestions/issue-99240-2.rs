enum Enum {
    Unit,
}
type Alias = Enum;

fn main() {
    Alias::
    Unit();
    //~^^ ERROR expected function, found enum variant `Alias::Unit`
}
