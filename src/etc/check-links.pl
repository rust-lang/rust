#!/usr/bin/perl -w

my $file = $ARGV[0];

my @lines = <>;

my $anchors = {};

my $i = 0;
foreach $line (@lines) {
    $i++;
    if ($line =~ m/id="([^"]+)"/) { 
        $anchors->{$1} = $i;
    }
}

$i = 0;
foreach $line (@lines) {
    $i++;
    while ($line =~ m/href="#([^"]+)"/g) { 
        if (! exists($anchors->{$1})) {
            print "$file:$i: $1 referenced\n";
        }
    }
}

