trait From<Src> {
    type Result;

    fn from(src: Src) -> Self::Result;
}

trait To {
    fn to<Dst>(
        self //~ ERROR the size for values of type
    ) -> <Dst as From<Self>>::Result where Dst: From<Self> { //~ ERROR the size for values of type
        From::from(self) //~ ERROR the size for values of type
    }
}

fn main() {}
