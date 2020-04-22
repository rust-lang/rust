// issue #4029
#[derive(Debug, Clone, Default Hash)]
struct S;

// issue #3898
#[derive(Debug, Clone, Default,, Hash)]
struct T;
