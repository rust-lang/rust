//@ edition:2018

macro_rules! foo_ { () => {}; }
use foo_ as foo;

macro_rules! bar { () => {}; }
