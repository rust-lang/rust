//@ edition: 2021

macro_rules! x {
    ($ty : item) => {};
}
x! {
  trait MyTrait { fn bar(c self) }
  //~^ ERROR expected identifier, found keyword `self`
  //~^^ ERROR expected one of `:`, `@`, or `|`, found keyword `self`
  //~^^^ ERROR expected one of `->`, `;`, `where`, or `{`, found `}`
}

fn main() {}
