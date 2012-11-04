#[allow(ctypes)];
#[allow(heap_memory)];
#[allow(implicit_copies)];
#[allow(managed_heap_memory)];
#[allow(non_camel_case_types)];
#[allow(non_implicitly_copyable_typarams)];
#[allow(owned_heap_memory)];
#[allow(path_statement)];
#[allow(structural_records)];
#[allow(unrecognized_lint)];
#[allow(unused_imports)];
#[allow(vecs_implicitly_copyable)];
#[allow(while_true)];

extern mod std;

fn print<T>(result: T) {
    io::println(fmt!("%?", result));
}
