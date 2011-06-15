

fn ret_int_i() -> int { ret 10; }

fn ret_ext_i() -> @int { ret @10; }

fn ret_int_tup() -> tup(int, int) { ret tup(10, 10); }

fn ret_ext_tup() -> @tup(int, int) { ret @tup(10, 10); }

fn ret_ext_mem() -> tup(@int, @int) { ret tup(@10, @10); }

fn ret_ext_ext_mem() -> @tup(@int, @int) { ret @tup(@10, @10); }

fn main() {
    let int int_i;
    let @int ext_i;
    let tup(int, int) int_tup;
    let @tup(int, int) ext_tup;
    let tup(@int, @int) ext_mem;
    let @tup(@int, @int) ext_ext_mem;
    int_i = ret_int_i(); // initializing

    int_i = ret_int_i(); // non-initializing

    int_i = ret_int_i(); // non-initializing

    ext_i = ret_ext_i(); // initializing

    ext_i = ret_ext_i(); // non-initializing

    ext_i = ret_ext_i(); // non-initializing

    int_tup = ret_int_tup(); // initializing

    int_tup = ret_int_tup(); // non-initializing

    int_tup = ret_int_tup(); // non-initializing

    ext_tup = ret_ext_tup(); // initializing

    ext_tup = ret_ext_tup(); // non-initializing

    ext_tup = ret_ext_tup(); // non-initializing

    ext_mem = ret_ext_mem(); // initializing

    ext_mem = ret_ext_mem(); // non-initializing

    ext_mem = ret_ext_mem(); // non-initializing

    ext_ext_mem = ret_ext_ext_mem(); // initializing

    ext_ext_mem = ret_ext_ext_mem(); // non-initializing

    ext_ext_mem = ret_ext_ext_mem(); // non-initializing

}