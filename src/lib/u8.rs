

fn add(u8 x, u8 y) -> u8 { ret x + y; }

fn sub(u8 x, u8 y) -> u8 { ret x - y; }

fn mul(u8 x, u8 y) -> u8 { ret x * y; }

fn div(u8 x, u8 y) -> u8 { ret x / y; }

fn rem(u8 x, u8 y) -> u8 { ret x % y; }

fn lt(u8 x, u8 y) -> bool { ret x < y; }

fn le(u8 x, u8 y) -> bool { ret x <= y; }

fn eq(u8 x, u8 y) -> bool { ret x == y; }

fn ne(u8 x, u8 y) -> bool { ret x != y; }

fn ge(u8 x, u8 y) -> bool { ret x >= y; }

fn gt(u8 x, u8 y) -> bool { ret x > y; }

iter range(u8 lo, u8 hi) -> u8 { while (lo < hi) { put lo; lo += 1u8; } }
// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C .. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
