mod stream {
    enum stream<T: send> { send(T, server::stream<T>), }
    mod server {
        impl<T: send> stream<T> {
            fn recv() -> extern fn(+stream<T>) -> stream::stream<T> {
              // resolve really should report just one error here.
              // Change the test case when it changes.
              fn recv(+pipe: stream<T>) -> stream::stream<T> { //~ ERROR attempt to use a type argument out of scope
                //~^ ERROR use of undeclared type name
                //~^^ ERROR attempt to use a type argument out of scope
                //~^^^ ERROR use of undeclared type name
                    option::unwrap(pipes::recv(pipe))
                }
                recv
            }
        }
        type stream<T: send> = pipes::recv_packet<stream::stream<T>>;
    }
}

fn main() {}
