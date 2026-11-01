// 2015
type DynPlain = dyn Path;
type DynRef = &dyn Path;
type DynLt = dyn 'a + Path;
type DynQuestion = dyn ?Path;
type DynFor = dyn for<'a> Path;
type DynParen = dyn(Path);
type Path = dyn::Path;
type Generic = dyn<Path>;
