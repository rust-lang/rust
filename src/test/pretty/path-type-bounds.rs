// pp-exact


trait Tr {
    fn dummy(&self) { }
}
impl Tr for isize { }

fn foo<'a>(x: Box<Tr + Sync + 'a>) -> Box<Tr + Sync + 'a> { x }

fn main() {
    let x: Box<Tr + Sync>;

    Box::new(1isize) as Box<Tr + Sync>;
}
