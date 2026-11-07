// rustfmt-imports_granularity: Item

use std::{
    fs,
    // (temporarily commented, we'll need this again in a second) io,
};

use foo::{
    self // this is important
};

use foo :: bar
;

use foo::{bar};

use foo::{
    bar
    // abc
};

use foo::{
    bar,
    // abc
};

use foo::{
    // 345
    bar
};

use foo::{
    self
    // abc
};

use foo::{
    self,
    // abc
};

use foo::{
    // 345
    self
};

use foo::{
    self // a
        ,
};

use foo::{ self /* a */ };

use foo::{ self /* a */, };

use foo::{
    // abc
    abc::{
        xyz
        // 123
    }
};

use foo::{
    bar,
    abc
};

use foo::{
    // abc
    bar,
    abc
};

use foo::{
    bar,
    // abc
    abc
};

use foo::{
    bar,
    abc
    // abc
};

use foo::{
    bar,
    abc,
    // abc
};

use foo::{
    self,
    // abc
    abc::{
        xyz
        // 123
    }
};

use foo::{
    self,
    // abc
    abc::{
        // 123
        xyz
    }
};

use path::{self /*comment*/,};
