macro_rules! number {
    (neg false, $self:ident) => { $self };
    ($signed:tt => $ty:ty;) => {
        number!(neg $signed, $self);
        //~^ ERROR no rules expected the token `$`
    };
}

number! { false => u8; }

fn main() {}
