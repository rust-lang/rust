#[derive(Copy(Bad))] //~ ERROR expected one of `)`, `,`, or `::`, found `(`
struct Test1;

#[derive(Copy="bad")] //~ ERROR expected one of `)`, `,`, or `::`, found `=`
struct Test2;

#[derive()] //~ WARNING empty trait list
struct Test3;

#[derive] //~ ERROR bad `derive` attribute
struct Test4;

fn main() {}
