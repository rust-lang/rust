

fn ret_int_i() -> int { ret 10; }

fn ret_ext_i() -> @int { ret @10; }

fn ret_int_rec() -> rec(int a, int b) { ret rec(a=10, b=10); }

fn ret_ext_rec() -> @rec(int a, int b) { ret @rec(a=10, b=10); }

fn ret_ext_mem() -> rec(@int a, @int b) { ret rec(a=@10, b=@10); }

fn ret_ext_ext_mem() -> @rec(@int a, @int b) { ret @rec(a=@10, b=@10); }

fn main() {
    let int int_i;
    let @int ext_i;
    let rec(int a, int b) int_rec;
    let @rec(int a, int b) ext_rec;
    let rec(@int a, @int b) ext_mem;
    let @rec(@int a, @int b) ext_ext_mem;
    int_i = ret_int_i(); // initializing

    int_i = ret_int_i(); // non-initializing

    int_i = ret_int_i(); // non-initializing

    ext_i = ret_ext_i(); // initializing

    ext_i = ret_ext_i(); // non-initializing

    ext_i = ret_ext_i(); // non-initializing

    int_rec = ret_int_rec(); // initializing

    int_rec = ret_int_rec(); // non-initializing

    int_rec = ret_int_rec(); // non-initializing

    ext_rec = ret_ext_rec(); // initializing

    ext_rec = ret_ext_rec(); // non-initializing

    ext_rec = ret_ext_rec(); // non-initializing

    ext_mem = ret_ext_mem(); // initializing

    ext_mem = ret_ext_mem(); // non-initializing

    ext_mem = ret_ext_mem(); // non-initializing

    ext_ext_mem = ret_ext_ext_mem(); // initializing

    ext_ext_mem = ret_ext_ext_mem(); // non-initializing

    ext_ext_mem = ret_ext_ext_mem(); // non-initializing

}