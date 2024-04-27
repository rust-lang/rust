fn foo<'a>(_: impl 'a Sized) {}
//~^ ERROR: expected `+` between lifetime and Sized
//~| ERROR: expected one of `:`, `@`, or `|`, found `)`
//~| ERROR: expected one of `)`, `+`, or `,`, found `Sized`
//~| ERROR: at least one trait must be specified

fn main(){
}
