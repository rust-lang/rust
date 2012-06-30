iface to_str { fn to_str() -> str; }

impl of to_str for int {
    fn to_str() -> str { int::str(self) }
}
impl of to_str for i8 {
    fn to_str() -> str { i8::str(self) }
}
impl of to_str for i16 {
    fn to_str() -> str { i16::str(self) }
}
impl of to_str for i32 {
    fn to_str() -> str { i32::str(self) }
}
impl of to_str for i64 {
    fn to_str() -> str { i64::str(self) }
}
impl of to_str for uint {
    fn to_str() -> str { uint::str(self) }
}
impl of to_str for u8 {
    fn to_str() -> str { u8::str(self) }
}
impl of to_str for u16 {
    fn to_str() -> str { u16::str(self) }
}
impl of to_str for u32 {
    fn to_str() -> str { u32::str(self) }
}
impl of to_str for u64 {
    fn to_str() -> str { u64::str(self) }
}
impl of to_str for float {
    fn to_str() -> str { float::to_str(self, 4u) }
}
impl of to_str for bool {
    fn to_str() -> str { bool::to_str(self) }
}
impl of to_str for () {
    fn to_str() -> str { "()" }
}
impl of to_str for str {
    fn to_str() -> str { self }
}

impl <A: to_str copy, B: to_str copy> of to_str for (A, B) {
    fn to_str() -> str {
        let (a, b) = self;
        "(" + a.to_str() + ", " + b.to_str() + ")"
    }
}
impl <A: to_str copy, B: to_str copy, C: to_str copy> of to_str for (A, B, C){
    fn to_str() -> str {
        let (a, b, c) = self;
        "(" + a.to_str() + ", " + b.to_str() + ", " + c.to_str() + ")"
    }
}

impl <A: to_str> of to_str for ~[A] {
    fn to_str() -> str {
        let mut acc = "[", first = true;
        for vec::each(self) |elt| {
            if first { first = false; }
            else { acc += ", "; }
            acc += elt.to_str();
        }
        acc += "]";
        acc
    }
}

impl <A: to_str> of to_str for @A {
    fn to_str() -> str { "@" + (*self).to_str() }
}
impl <A: to_str> of to_str for ~A {
    fn to_str() -> str { "~" + (*self).to_str() }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_simple_types() {
        assert 1.to_str() == "1";
        assert (-1).to_str() == "-1";
        assert 200u.to_str() == "200";
        assert 2u8.to_str() == "2";
        assert true.to_str() == "true";
        assert false.to_str() == "false";
        assert ().to_str() == "()";
        assert "hi".to_str() == "hi";
    }

    #[test]
    fn test_tuple_types() {
        assert (1, 2).to_str() == "(1, 2)";
        assert ("a", "b", false).to_str() == "(a, b, false)";
        assert ((), ((), 100)).to_str() == "((), ((), 100))";
    }

    fn test_vectors() {
        let x: ~[int] = ~[];
        assert x.to_str() == "~[]";
        assert (~[1]).to_str() == "~[1]";
        assert (~[1, 2, 3]).to_str() == "~[1, 2, 3]";
        assert (~[~[], ~[1], ~[1, 1]]).to_str() ==
               "~[~[], ~[1], ~[1, 1]]";
    }

    fn test_pointer_types() {
        assert (@1).to_str() == "@1";
        assert (~(true, false)).to_str() == "~(true, false)";
    }
}
