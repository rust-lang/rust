enum ast<'a> {
    num(usize),
    add(&'a ast<'a>, &'a ast<'a>)
}

fn build() {
    let x = ast::num(3);
    let y = ast::num(4);
    let z = ast::add(&x, &y);
    compute(&z);
}

fn compute(x: &ast) -> usize {
    match *x {
      ast::num(x) => { x }
      ast::add(x, y) => { compute(x) + compute(y) }
    }
}

fn map_nums<'a,'b, F>(x: &ast, f: &mut F) -> &'a ast<'b> where F: FnMut(usize) -> usize {
    match *x {
      ast::num(x) => {
        return &ast::num((*f)(x)); //~ ERROR borrowed value does not live long enough
      }
      ast::add(x, y) => {
        let m_x = map_nums(x, f);
        let m_y = map_nums(y, f);
        return &ast::add(m_x, m_y);  //~ ERROR borrowed value does not live long enough
      }
    }
}

fn main() {}
