struct T1<const X1: usize>;
struct T2<const X1: usize, const X2: usize>;
struct T3<const X1: usize, const X2: usize, const X3: usize>;

impl T1<1> {
    const C: () = ();
}

impl T2<1, 2> {
    const C: () = ();
}

impl T3<1, 2, 3> {
    const C: () = ();
}

fn main() {
    T1<1>::C; //~ ERROR: comparison operators cannot be chained
    T2<1, 2>::C; //~ ERROR: expected one of `.`, `;`, `?`, `}`, or an operator, found `,`
    T3<1, 2, 3>::C; //~ ERROR: expected one of `.`, `;`, `?`, `}`, or an operator, found `,`
}
