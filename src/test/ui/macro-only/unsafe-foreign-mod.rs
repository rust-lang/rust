// check-fail

unsafe extern "C++" { //~ERROR extern block cannot be declared unsafe
                      //~|ERROR invalid ABI
}

fn main() {}
