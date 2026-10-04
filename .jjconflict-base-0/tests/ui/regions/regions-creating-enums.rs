enum Ast<'a> {
    Num(usize),
    Add(&'a Ast<'a>, &'a Ast<'a>)
}

fn build() {
    let x = Ast::Num(3);
    let y = Ast::Num(4);
    let z = Ast::Add(&x, &y);
    compute(&z);
}

fn compute(x: &Ast) -> usize {
    match *x {
      Ast::Num(x) => { x }
      Ast::Add(x, y) => { compute(x) + compute(y) }
    }
}

fn map_nums<'a,'b, F>(x: &Ast, f: &mut F) -> &'a Ast<'b> where F: FnMut(usize) -> usize {
    match *x {
      Ast::Num(x) => {
        return &Ast::Num((*f)(x)); //~ ERROR cannot return reference to temporary value
      }
      Ast::Add(x, y) => {
        let m_x = map_nums(x, f);
        let m_y = map_nums(y, f);
        return &Ast::Add(m_x, m_y);  //~ ERROR cannot return reference to temporary value
      }
    }
}

fn main() {}
