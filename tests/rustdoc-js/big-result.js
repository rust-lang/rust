// exact-check

const EXPECTED = [
    {
        'query': 'First',
        'in_args': (function() {
            // Generate the list of 200 items that should match.
            const results = [];
            function generate(lx, ly) {
                for (const x of lx) {
                    for (const y of ly) {
                        results.push({
                            'path': `big_result::${y}`,
                            'name': x,
                        });
                    }
                }
            }
            // Fewest parameters that still match go on top.
            generate(
                ['u', 'v', 'w', 'x', 'y'],
                ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
            );
            generate(
                ['p', 'q', 'r', 's', 't'],
                ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
            );
            generate(
                ['k', 'l', 'm', 'n', 'o'],
                ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
            );
            generate(
                ['f', 'g', 'h', 'i', 'j'],
                ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
            );
            return results;
        })(),
    },
];
