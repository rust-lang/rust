extern crate minibevy;
extern crate minirapier;

use minibevy::Resource;
use minirapier::Ray;

fn insert_resource<R: Resource>(_resource: R) {}

struct Res;
impl Resource for Res {}

fn main() {
    insert_resource(Res.into());
}
