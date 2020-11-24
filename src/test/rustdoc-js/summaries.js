// ignore-tidy-linelength

const QUERY = ['summaries', 'summaries::Sidebar', 'summaries::Sidebar2'];

const EXPECTED = [
    {
        'others': [
           { 'path': '', 'name': 'summaries', 'desc': 'This <em>summary</em> has a link and <code>code</code>.' },
        ],
    },
    {
        'others': [
            { 'path': 'summaries', 'name': 'Sidebar', 'desc': 'This <code>code</code> will be rendered in a code tag.' },
        ],
    },
    {
        'others': [
            { 'path': 'summaries', 'name': 'Sidebar2', 'desc': '' },
        ],
    },
];
