enum ast/& {
    num(uint),
    add(&ast, &ast)
}

fn build() {
    let x = num(3u);
    let y = num(4u);
    let z = add(&x, &y);
    compute(&z);
}

fn compute(x: &ast) -> uint {
    alt *x {
      num(x) { x }
      add(x, y) { compute(x) + compute(y) }
    }
}

fn map_nums(x: &ast, f: fn(uint) -> uint) -> &ast {
    alt *x {
      num(x) {
        ret &num(f(x)); //~ ERROR mismatched types: expected `&ast/&` but found
      }
      add(x, y) {
        let m_x = map_nums(x, f);
        let m_y = map_nums(y, f);
        ret &add(m_x, m_y);  //~ ERROR mismatched types: expected `&ast/&` but found
      }
    }
}

fn main() {}