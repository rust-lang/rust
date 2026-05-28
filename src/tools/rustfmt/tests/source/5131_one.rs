// rustfmt-imports_granularity: One

pub use foo::x;
pub use foo::x as x2;
pub use foo::y;
use bar::a;
use bar::b;
use bar::b::f;
use bar::b::f as f2;
use bar::b::g;
use bar::c;
use bar::d::e;
use bar::d::e as e2;
use qux::h;
use qux::i;
