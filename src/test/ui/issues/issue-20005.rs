trait From<Src> {
    type Result;

    fn from(src: Src) -> Self::Result;
}

trait To {
    fn to<Dst>(  //~ ERROR the size for values of type
        self
    ) -> <Dst as From<Self>>::Result where Dst: From<Self> {
        From::from(self)
    }
}

fn main() {}
