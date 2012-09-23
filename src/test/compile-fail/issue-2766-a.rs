mod stream {
    #[legacy_exports];
    enum stream<T: Send> { send(T, server::stream<T>), }
    mod server {
        #[legacy_exports];
        impl<T: Send> stream<T> {
            fn recv() -> extern fn(+v: stream<T>) -> stream::stream<T> {
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
        type stream<T: Send> = pipes::RecvPacket<stream::stream<T>>;
    }
}

fn main() {}
