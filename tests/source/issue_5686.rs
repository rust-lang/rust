#[repr(u8)]
enum MyEnum {
    UnitWithExplicitDiscriminant = 0,
    EmptyStructSingleLineBlockComment {
        /* Comment */
    } = 1,
    EmptyStructMultiLineBlockComment {
        /*
         * Comment
         */
    } = 2,
    EmptyStructLineComment {
        // comment
    } = 3,
    EmptyTupleSingleLineBlockComment(
        /* Comment */
    ) = 4,
    EmptyTupleMultiLineBlockComment(
        /*
         * Comment
         */
    ) = 5,
    EmptyTupleLineComment(
        // comment
    ) = 6,
}

enum Animal {
    Dog(/* tuple variant closer in comment -> ) */) = 1,
    #[hello(world)]
    Cat(/* tuple variant close in leading attribute */) = 2,
    Bee(/* tuple variant closer on associated field attribute */ #[hello(world)] usize) = 3,
    Fox(/* tuple variant closer on const fn call */) = some_const_fn(),
    Ant(/* tuple variant closer on macro call */) = some_macro!(),
    Snake {/* stuct variant closer in comment -> } */} = 6,
    #[hell{world}]
    Cobra {/* struct variant close in leading attribute */} = 6,
    Eagle {/* struct variant closer on associated field attribute */ #[hell{world}]value: Sting} = 7,
    Koala {/* struct variant closer on macro call */} = some_macro!{}
}
