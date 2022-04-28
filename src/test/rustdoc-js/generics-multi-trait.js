// exact-check

const QUERY = [
    'Result<SomeTrait>',
    'Zzzzzzzzzzzzzzzzzz',
    'Nonononononononono',
];

const EXPECTED = [
    // check one of the generic items
    {
        'in_args': [],
        'returned': [],
    },
    {
        'in_args': [],
        'returned': [],
    },
    // ignore the name of the generic itself
    {
        'in_args': [],
        'returned': [],
    },
];
