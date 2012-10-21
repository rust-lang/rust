mod stream {
    #[legacy_exports];
    enum Stream<T: Send> { send(T, server::Stream<T>), }
    mod server {
        #[legacy_exports];
        impl<T: Send> Stream<T> {
            fn recv() -> extern fn(+v: Stream<T>) -> stream::Stream<T> {
              // resolve really should report just one error here.
              // Change the test case when it changes.
              fn recv(+pipe: Stream<T>) -> stream::Stream<T> { //~ ERROR attempt to use a type argument out of scope
                //~^ ERROR use of undeclared type name
                //~^^ ERROR attempt to use a type argument out of scope
                //~^^^ ERROR use of undeclared type name
                    option::unwrap(pipes::recv(pipe))
                }
                recv
            }
        }
        type Stream<T: Send> = pipes::RecvPacket<stream::Stream<T>>;
    }
}

fn main() {}
