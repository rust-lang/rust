
type PrivateTest<'a, I> = (Box<Parser<Input=I, Output=char> + 'a>, Box<Parser<Input=I, Output=char> + 'a>);

pub type PublicTest<'a, I, O> = Result<Vec<MyLongType>, Box<Parser<Input=I, Output=char> + 'a>, Box<Parser<Input=I, Output=char> + 'a>>;

pub type LongGenericListTest<'a, 'b, 'c, 'd, LONGPARAMETERNAME, LONGPARAMETERNAME, LONGPARAMETERNAME, A, B, C> = Option<Vec<MyType>>;

pub type Exactly100CharsTest<'a, 'b, 'c, 'd, LONGPARAMETERNAME, LONGPARAMETERNAME, A, B> = Vec<i32>;

pub type Exactly101CharsTest<'a, 'b, 'c, 'd, LONGPARAMETERNAME, LONGPARAMETERNAME, A, B> = Vec<Test>;

pub type CommentTest< /* Lifetime */ 'a
            ,
        // Type
        T
                    > = ();
