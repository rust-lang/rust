trait connection {
    fn read() -> int;
}

trait connection_factory<C: connection> {
    fn create() -> C;
}

type my_connection = ();
type my_connection_factory = ();

impl (): connection {
    fn read() -> int { 43 }
}

impl my_connection_factory: connection_factory<my_connection> {
    fn create() -> my_connection { () }
}

fn main() {
    let factory = ();
    let connection = factory.create();
    let result = connection.read();
    assert result == 43;
}
