trait connection {
    fn read() -> int;
}

trait connection_factory<C: connection> {
    fn create() -> C;
}

type my_connection = ();
type my_connection_factory = ();

impl of connection for () {
    fn read() -> int { 43 }
}

impl of connection_factory<my_connection> for my_connection_factory {
    fn create() -> my_connection { () }
}

fn main() {
    let factory = ();
    let connection = factory.create();
    let result = connection.read();
    assert result == 43;
}
