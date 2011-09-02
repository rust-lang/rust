

pure fn add(x: u8, y: u8) -> u8 { ret x + y; }

pure fn sub(x: u8, y: u8) -> u8 { ret x - y; }

pure fn mul(x: u8, y: u8) -> u8 { ret x * y; }

pure fn div(x: u8, y: u8) -> u8 { ret x / y; }

pure fn rem(x: u8, y: u8) -> u8 { ret x % y; }

pure fn lt(x: u8, y: u8) -> bool { ret x < y; }

pure fn le(x: u8, y: u8) -> bool { ret x <= y; }

pure fn eq(x: u8, y: u8) -> bool { ret x == y; }

pure fn ne(x: u8, y: u8) -> bool { ret x != y; }

pure fn ge(x: u8, y: u8) -> bool { ret x >= y; }

pure fn gt(x: u8, y: u8) -> bool { ret x > y; }

iter range(lo: u8, hi: u8) -> u8 { while lo < hi { put lo; lo += 1u8; } }
// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
