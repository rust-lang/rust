#!/usr/bin/perl -w
# Copyright 2014 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

my $file = $ARGV[0];

my @lines = <>;

my $anchors = {};

my $i = 0;
for $line (@lines) {
    $i++;
    if ($line =~ m/id="([^"]+)"/) {
        $anchors->{$1} = $i;
    }
}

$i = 0;
for $line (@lines) {
    $i++;
    while ($line =~ m/href="#([^"]+)"/g) {
        if (! exists($anchors->{$1})) {
            print "$file:$i: $1 referenced\n";
        }
    }
}
