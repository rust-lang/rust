impl ! {} // OK
impl ! where u8: Copy {} // OK

impl Trait Type {} //~ ERROR missing `for` in a trait impl
impl ?Sized for Type {} //~ ERROR expected a trait, found type
