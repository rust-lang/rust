// edition:2018

type A0 = dyn;
type A1 = dyn::dyn; //~ERROR expected identifier, found reserved keyword
type A2 = dyn<dyn, dyn>; //~ERROR expected identifier, found `<`
type A3 = dyn<<dyn as dyn>::dyn>;

fn main() {}
