trait Foo {}

trait Bar {}

impl<'a> Foo for Bar + 'a {}
