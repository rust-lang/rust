// ICE Where clause `Binder(..)` was applicable to `Obligation(..)` but now is not
// issue: rust-lang/rust#84727

struct Cell<Fg, Bg = Fg> {
    foreground: Color<Fg>,
    //~^ ERROR cannot find type `Color` in this scope
    background: Color<Bg>,
    //~^ ERROR cannot find type `Color` in this scope
}

trait Over<Bottom, Output> {
    fn over(self) -> Output;
}

impl<TopFg, TopBg, BottomFg, BottomBg, NewFg, NewBg>
    Over<Cell<BottomFg, BottomBg>, Cell<NewFg, NewBg>> for Cell<TopFg, TopBg>
where
    Self: Over<Color<BottomBg>, Cell<NewFg>>,
    //~^ ERROR cannot find type `Color` in this scope
{
    fn over(self) -> Cell<NewFg> {
    //~^ ERROR mismatched types
        self.over();
    }
}

impl<'b, TopFg, TopBg, BottomFg, BottomBg> Over<&Cell<BottomFg, BottomBg>, ()>
    for Cell<TopFg, TopBg>
where
    Cell<TopFg, TopBg>: Over<Cell<BottomFg>, Cell<BottomFg>>,
{
    fn over(self) -> Cell<NewBg> {
    //~^ ERROR cannot find type `NewBg` in this scope
        self.over();
    }
}

pub fn main() {}
