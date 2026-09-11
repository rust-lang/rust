type Foo
// comment1
// interlinear1
where
    // comment2
    // interlinear2
    A: B, // comment3
    C: D, // comment4
// interlinear3
= E; // comment5

type Foo
// comment6
// interlinear4
where
    // comment7
    // interlinear5
    A: B, // comment8
    C: D, // comment9
// interlinear6
= E
// comment10
// interlinear7
where
    // comment11
    // interlinear8
    F: G, // comment12
    H: I; // comment13

type Foo // comment14
    // interlinear9
    = E
// comment15
// interlinear10
where
    // comment16
    // interlinear11
    F: G, // comment17
    H: I; // comment18
