// ignore-tidy-linelength

const EXPECTED = [
    {
        'query': 'summaries',
        'others': [
           { 'path': '', 'name': 'summaries', 'desc': 'This <em>summary</em> has a link, [<code>code</code>], and <code>Sidebar2</code> intra-doc.' },
        ],
    },
    {
        'query': 'summaries::Sidebar',
        'others': [
            { 'path': 'summaries', 'name': 'Sidebar', 'desc': 'This <code>code</code> will be rendered in a code tag.' },
        ],
    },
    {
        'query': 'summaries::Sidebar2',
        'others': [
            { 'path': 'summaries', 'name': 'Sidebar2', 'desc': '' },
        ],
    },
];
