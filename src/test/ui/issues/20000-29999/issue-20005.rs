trait From<Src> {
    type Result;

    fn from(src: Src) -> Self::Result;
}

trait To {
    fn to<Dst>(
        self
    ) -> <Dst as From<Self>>::Result where Dst: From<Self> { //~ ERROR the size for values of type
        From::from(self)
    }
}

fn main() {}
