// Testing types for the serializer.  This should be made more formal.

enum test1 {
    t1_a(int), t1_b(str)
}

type test2 = {
    f: int, g: str
};

enum test3 {
    t3_a, t3_b
}

enum test4 {
    t4_a(test1), t4_b(test2), t4_c(@test2), t4_d(@test4)
}

type spanned<A> = {
    node: A,
    span: { lo: uint, hi: uint }
};

type test5 = {
    s1: spanned<test4>,
    s2: spanned<uint>
};

type test6 = option<int>;

fn main() {}