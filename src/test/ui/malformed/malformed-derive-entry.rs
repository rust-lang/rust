#[derive(Copy(Bad))] //~ ERROR expected one of `)`, `,`, or `::`, found `(`
struct Test1;

#[derive(Copy="bad")] //~ ERROR expected one of `)`, `,`, or `::`, found `=`
struct Test2;

#[derive()] //~ WARNING empty trait list
struct Test3;

#[derive] //~ ERROR attribute must be of the form `#[derive(Trait1, Trait2, ...)]`
struct Test4;

fn main() {}
