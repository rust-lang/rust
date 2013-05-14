#!/usr/bin/perl

#
# This is a tool that helps with debugging incorrect monomorphic instance collapse.
#
# To use:
#    $ RUST_LOG=rustc::middle::trans::monomorphize rustc ARGS 2>&1 >log.txt
#    $ ./monodebug.pl log.txt
#
# This will show all generics that got collapsed. You can inspect this list to find the instances
# that were mistakenly combined into one. Fixes will (most likely) be applied to type_use.rs.
#
# Questions about this tool go to pcwalton.
#

use strict;
use warnings;
use Data::Dumper qw(Dumper);
use Text::Balanced qw(extract_bracketed);

my %funcs;
while (<>) {
    chomp;
    /^rust: ~"monomorphic_fn\((.*)"$/ or next;
    my $text = $1;
    $text =~ /fn_id=(\{ crate: \d+, node: \d+ \} \([^)]+\)), real_substs=(.*?), substs=(.*?), hash_id = \@\{ (.*) \}$/ or next;
    my ($fn_id, $real_substs, $substs, $hash_id) = ($1, $2, $3, $4);

    #print "$hash_id\n";
    $hash_id =~ /^def: { crate: \d+, node: \d+ }, params: ~\[ (.*) \], impl_did_opt: (?:None|Some\({ crate: \d+, node: \d+ }\))$/ or next;
    my $params = $1;

    my @real_substs;
    @real_substs = $real_substs =~ /\\"(.*?)\\"/g;

    my @mono_params;
    while (1) {
        $params =~ s/^, //;
        if ($params =~ s/^mono_precise//) {
            extract_bracketed($params, '()');
            push @mono_params, 'precise';
            next;
        }
        if ($params =~ s/^mono_repr//) {
            my $sub = extract_bracketed($params, '()');
            push @mono_params, "repr($sub)";
            next;
        }
        if ($params =~ s/^mono_any//) {
            push @mono_params, "any";
            next;
        }
        last;
    }

    my @key_params;
    for (my $i = 0; $i < @mono_params; ++$i) {
        if ($mono_params[$i] eq 'precise') {
            push @key_params, 'precise(' . $real_substs[$i] . ')';
        } else {
            push @key_params, $mono_params[$i];
        }
    }

    my $key = "$fn_id with " . (join ', ', @key_params);
    $funcs{$key}{$real_substs} = 1;
}

while (my ($key, $substs) = each %funcs) {
    my @params = keys %$substs;
    next if @params == 1;

    print "$key\n";
    print(('-' x (length $key)), $/);
    for my $param (@params) {
        print "$param\n";
    }
    print "\n";
}
