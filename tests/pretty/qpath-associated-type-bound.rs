// pp-exact


mod m {
    pub trait Tr {
        type Ts: super::Tu;
    }
}

trait Tu {
    fn dummy() {}
}

fn foo<T: m::Tr>() { <T as m::Tr>::Ts::dummy(); }

fn main() {}
